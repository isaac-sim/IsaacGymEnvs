# Copyright (c) 2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""IndustReal: algorithms module.

Contains functions that implement Simulation-Aware Policy Update (SAPU), SDF-Based Reward, and Sampling-Based Curriculum (SBC).

Not intended to be executed as a standalone script.
"""

import numpy as np
from pysdf import SDF
import torch
import trimesh
from urdfpy import URDF
import warp as wp


"""
Simulation-Aware Policy Update (SAPU)
"""


def load_asset_mesh_in_warp(urdf_path, sample_points, num_samples, device):
    """Create mesh object in Warp."""

    urdf = URDF.load(urdf_path)
    mesh = urdf.links[0].collision_mesh

    wp_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(mesh.faces.flatten(), dtype=wp.int32, device=device),
    )

    if sample_points:
        # Sample points on surface of mesh
        sampled_points, _ = trimesh.sample.sample_surface_even(mesh, num_samples)
        wp_mesh_sampled_points = wp.array(sampled_points, dtype=wp.vec3, device=device)
        return wp_mesh, wp_mesh_sampled_points
    else:
        return wp_mesh


def load_asset_meshes_in_warp(plug_files, socket_files, num_samples, device):
    """Create mesh objects in Warp for all environments."""

    # Load and store plug meshes and (if desired) sampled points
    plug_meshes, plug_meshes_sampled_points = [], []
    for i in range(len(plug_files)):
        plug_mesh, sampled_points = load_asset_mesh_in_warp(
            urdf_path=plug_files[i],
            sample_points=True,
            num_samples=num_samples,
            device=device,
        )
        plug_meshes.append(plug_mesh)
        plug_meshes_sampled_points.append(sampled_points)

    # Load and store socket meshes
    socket_meshes = [
        load_asset_mesh_in_warp(
            urdf_path=socket_files[i],
            sample_points=False,
            num_samples=-1,
            device=device,
        )
        for i in range(len(socket_files))
    ]

    return plug_meshes, plug_meshes_sampled_points, socket_meshes


def get_max_interpen_dists(
    asset_indices,
    plug_pos,
    plug_quat,
    socket_pos,
    socket_quat,
    wp_plug_meshes_sampled_points,
    wp_socket_meshes,
    wp_device,
    device,
):
    """Get maximum interpenetration distances between plugs and sockets."""

    num_envs = len(plug_pos)
    max_interpen_dists = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    for i in range(num_envs):
        asset_idx = asset_indices[i]

        # Compute transform from plug frame to socket frame
        plug_transform = wp.transform(plug_pos[i], plug_quat[i])
        socket_transform = wp.transform(socket_pos[i], socket_quat[i])
        socket_inv_transform = wp.transform_inverse(socket_transform)
        plug_to_socket_transform = wp.transform_multiply(
            plug_transform, socket_inv_transform
        )

        # Transform plug mesh vertices to socket frame
        plug_points = wp.clone(wp_plug_meshes_sampled_points[asset_idx])
        wp.launch(
            kernel=transform_points,
            dim=len(plug_points),
            inputs=[plug_points, plug_points, plug_to_socket_transform],
            device=wp_device,
        )

        # Compute max interpenetration distance between plug and socket
        interpen_dist_plug_socket = wp.zeros(
            (len(plug_points),), dtype=wp.float32, device=wp_device
        )
        wp.launch(
            kernel=get_interpen_dist,
            dim=len(plug_points),
            inputs=[
                plug_points,
                wp_socket_meshes[asset_idx].id,
                interpen_dist_plug_socket,
            ],
            device=wp_device,
        )

        max_interpen_dist = -torch.min(wp.to_torch(interpen_dist_plug_socket))

        # Store interpenetration flag and max interpenetration distance
        if max_interpen_dist > 0.0:
            max_interpen_dists[i] = max_interpen_dist

    return max_interpen_dists


def get_sapu_reward_scale(
    asset_indices,
    plug_pos,
    plug_quat,
    socket_pos,
    socket_quat,
    wp_plug_meshes_sampled_points,
    wp_socket_meshes,
    interpen_thresh,
    wp_device,
    device,
):
    """Compute reward scale for SAPU."""

    # Get max interpenetration distances
    max_interpen_dists = get_max_interpen_dists(
        asset_indices=asset_indices,
        plug_pos=plug_pos,
        plug_quat=plug_quat,
        socket_pos=socket_pos,
        socket_quat=socket_quat,
        wp_plug_meshes_sampled_points=wp_plug_meshes_sampled_points,
        wp_socket_meshes=wp_socket_meshes,
        wp_device=wp_device,
        device=device,
    )

    # Determine if envs have low interpenetration or high interpenetration
    low_interpen_envs = torch.nonzero(max_interpen_dists <= interpen_thresh)
    high_interpen_envs = torch.nonzero(max_interpen_dists > interpen_thresh)

    # Compute reward scale
    reward_scale = 1 - torch.tanh(
        max_interpen_dists[low_interpen_envs] / interpen_thresh
    )

    return low_interpen_envs, high_interpen_envs, reward_scale


"""
SDF-Based Reward
"""


def get_plug_goal_sdfs(
    wp_plug_meshes, asset_indices, socket_pos, socket_quat, wp_device
):
    """Get SDFs of plug meshes at goal pose."""

    num_envs = len(socket_pos)
    plug_goal_sdfs = []

    for i in range(num_envs):
        # Create copy of plug mesh
        mesh = wp_plug_meshes[asset_indices[i]]
        mesh_points = wp.clone(mesh.points)
        mesh_indices = wp.clone(mesh.indices)
        mesh_copy = wp.Mesh(points=mesh_points, indices=mesh_indices)

        # Transform plug mesh from current pose to goal pose
        # NOTE: In source OBJ files, when plug and socket are assembled,
        # their poses are identical
        goal_transform = wp.transform(socket_pos[i], socket_quat[i])
        wp.launch(
            kernel=transform_points,
            dim=len(mesh_copy.points),
            inputs=[mesh_copy.points, mesh_copy.points, goal_transform],
            device=wp_device,
        )

        # Rebuild BVH (see https://nvidia.github.io/warp/_build/html/modules/runtime.html#meshes)
        mesh_copy.refit()

        # Create SDF from transformed mesh
        sdf = SDF(mesh_copy.points.numpy(), mesh_copy.indices.numpy().reshape(-1, 3))

        plug_goal_sdfs.append(sdf)

    return plug_goal_sdfs


def get_sdf_reward(
    wp_plug_meshes_sampled_points,
    asset_indices,
    plug_pos,
    plug_quat,
    plug_goal_sdfs,
    wp_device,
    device,
):
    """Calculate SDF-based reward."""

    num_envs = len(plug_pos)
    sdf_reward = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    for i in range(num_envs):
        # Create copy of sampled points
        sampled_points = wp.clone(wp_plug_meshes_sampled_points[asset_indices[i]])

        # Transform sampled points from original plug pose to current plug pose
        curr_transform = wp.transform(plug_pos[i], plug_quat[i])
        wp.launch(
            kernel=transform_points,
            dim=len(sampled_points),
            inputs=[sampled_points, sampled_points, curr_transform],
            device=wp_device,
        )

        # Get SDF values at transformed points
        sdf_dists = torch.from_numpy(plug_goal_sdfs[i](sampled_points.numpy())).double()

        # Clamp values outside isosurface and take absolute value
        sdf_dists = torch.abs(torch.where(sdf_dists > 0.0, 0.0, sdf_dists))

        sdf_reward[i] = torch.mean(sdf_dists)

    sdf_reward = -torch.log(sdf_reward)

    return sdf_reward


"""
Sampling-Based Curriculum (SBC)
"""


def get_curriculum_reward_scale(cfg_task, curr_max_disp):
    """Compute reward scale for SBC."""

    # Compute difference between max downward displacement at beginning of training (easiest condition)
    # and current max downward displacement (based on current curriculum stage)
    # NOTE: This number increases as curriculum gets harder
    curr_stage_diff = cfg_task.rl.curriculum_height_bound[1] - curr_max_disp

    # Compute difference between max downward displacement at beginning of training (easiest condition)
    # and min downward displacement (hardest condition)
    final_stage_diff = (
        cfg_task.rl.curriculum_height_bound[1] - cfg_task.rl.curriculum_height_bound[0]
    )

    # Compute reward scale
    reward_scale = curr_stage_diff / final_stage_diff + 1.0

    return reward_scale


def get_new_max_disp(curr_success, cfg_task, curr_max_disp):
    """Update max downward displacement of plug at beginning of episode, based on success rate."""

    if curr_success > cfg_task.rl.curriculum_success_thresh:
        # If success rate is above threshold, reduce max downward displacement until min value
        # NOTE: height_step[0] is negative
        new_max_disp = max(
            curr_max_disp + cfg_task.rl.curriculum_height_step[0],
            cfg_task.rl.curriculum_height_bound[0],
        )

    elif curr_success < cfg_task.rl.curriculum_failure_thresh:
        # If success rate is below threshold, increase max downward displacement until max value
        # NOTE: height_step[1] is positive
        new_max_disp = min(
            curr_max_disp + cfg_task.rl.curriculum_height_step[1],
            cfg_task.rl.curriculum_height_bound[1],
        )  

    else:
        # Maintain current max downward displacement
        new_max_disp = curr_max_disp

    return new_max_disp


"""
Bonus and Success Checking
"""


def get_keypoint_offsets(num_keypoints, device):
    """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = (
        torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    )

    return keypoint_offsets


def check_plug_close_to_socket(
    keypoints_plug, keypoints_socket, dist_threshold, progress_buf
):
    """Check if plug is close to socket."""

    # Compute keypoint distance between plug and socket
    keypoint_dist = torch.norm(keypoints_socket - keypoints_plug, p=2, dim=-1)

    # Check if keypoint distance is below threshold
    is_plug_close_to_socket = torch.where(
        torch.sum(keypoint_dist, dim=-1) < dist_threshold,
        torch.ones_like(progress_buf),
        torch.zeros_like(progress_buf),
    )

    return is_plug_close_to_socket


def check_plug_engaged_w_socket(
    plug_pos, socket_top_pos, keypoints_plug, keypoints_socket, cfg_task, progress_buf
):
    """Check if plug is engaged with socket."""

    # Check if base of plug is below top of socket
    # NOTE: In assembled state, plug origin is coincident with socket origin;
    # thus plug pos must be offset to compute actual pos of base of plug
    is_plug_below_engagement_height = (
        plug_pos[:, 2] + cfg_task.env.socket_base_height < socket_top_pos[:, 2]
    )

    # Check if plug is close to socket
    # NOTE: This check addresses edge case where base of plug is below top of socket,
    # but plug is outside socket
    is_plug_close_to_socket = check_plug_close_to_socket(
        keypoints_plug=keypoints_plug,
        keypoints_socket=keypoints_socket,
        dist_threshold=cfg_task.rl.close_error_thresh,
        progress_buf=progress_buf,
    )

    # Combine both checks
    is_plug_engaged_w_socket = torch.logical_and(
        is_plug_below_engagement_height, is_plug_close_to_socket
    )

    return is_plug_engaged_w_socket


def check_plug_inserted_in_socket(
    plug_pos, socket_pos, keypoints_plug, keypoints_socket, cfg_task, progress_buf
):
    """Check if plug is inserted in socket."""

    # Check if plug is within threshold distance of assembled state
    is_plug_below_insertion_height = (
        plug_pos[:, 2] < socket_pos[:, 2] + cfg_task.rl.success_height_thresh
    )

    # Check if plug is close to socket
    # NOTE: This check addresses edge case where plug is within threshold distance of
    # assembled state, but plug is outside socket
    is_plug_close_to_socket = check_plug_close_to_socket(
        keypoints_plug=keypoints_plug,
        keypoints_socket=keypoints_socket,
        dist_threshold=cfg_task.rl.close_error_thresh,
        progress_buf=progress_buf,
    )

    # Combine both checks
    is_plug_inserted_in_socket = torch.logical_and(
        is_plug_below_insertion_height, is_plug_close_to_socket
    )

    return is_plug_inserted_in_socket


def check_gear_engaged_w_shaft(
    keypoints_gear,
    keypoints_shaft,
    gear_pos,
    shaft_pos,
    asset_info_gears,
    cfg_task,
    progress_buf,
):
    """Check if gear is engaged with shaft."""

    # Check if bottom of gear is below top of shaft
    is_gear_below_engagement_height = (
        gear_pos[:, 2]
        < shaft_pos[:, 2]
        + asset_info_gears.base.height
        + asset_info_gears.shafts.height
    )

    # Check if gear is close to shaft
    # Note: This check addresses edge case where gear is within threshold distance of
    # assembled state, but gear is outside shaft
    is_gear_close_to_shaft = check_plug_close_to_socket(
        keypoints_plug=keypoints_gear,
        keypoints_socket=keypoints_shaft,
        dist_threshold=cfg_task.rl.close_error_thresh,
        progress_buf=progress_buf,
    )

    # Combine both checks
    is_gear_engaged_w_shaft = torch.logical_and(
        is_gear_below_engagement_height, is_gear_close_to_shaft
    )

    return is_gear_engaged_w_shaft


def check_gear_inserted_on_shaft(
    gear_pos, shaft_pos, keypoints_gear, keypoints_shaft, cfg_task, progress_buf
):
    """Check if gear is inserted on shaft."""

    # Check if gear is within threshold distance of assembled state
    is_gear_below_insertion_height = (
        gear_pos[:, 2] < shaft_pos[:, 2] + cfg_task.rl.success_height_thresh
    )

    # Check if keypoint distance is below threshold
    is_gear_close_to_shaft = check_plug_close_to_socket(
        keypoints_plug=keypoints_gear,
        keypoints_socket=keypoints_shaft,
        dist_threshold=cfg_task.rl.close_error_thresh,
        progress_buf=progress_buf,
    )

    # Combine both checks
    is_gear_inserted_on_shaft = torch.logical_and(
        is_gear_below_insertion_height, is_gear_close_to_shaft
    )

    return is_gear_inserted_on_shaft


def get_engagement_reward_scale(
    plug_pos, socket_pos, is_plug_engaged_w_socket, success_height_thresh, device
):
    """Compute scale on reward. If plug is not engaged with socket, scale is zero.
    If plug is engaged, scale is proportional to distance between plug and bottom of socket."""

    # Set default value of scale to zero
    num_envs = len(plug_pos)
    reward_scale = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    # For envs in which plug and socket are engaged, compute positive scale
    engaged_idx = np.argwhere(is_plug_engaged_w_socket.cpu().numpy().copy()).squeeze()
    height_dist = plug_pos[engaged_idx, 2] - socket_pos[engaged_idx, 2]
    # NOTE: Edge case: if success_height_thresh is greater than 0.1,
    # denominator could be negative
    reward_scale[engaged_idx] = 1.0 / ((height_dist - success_height_thresh) + 0.1)

    return reward_scale


"""
Warp Kernels
"""


# Transform points from source coordinate frame to destination coordinate frame
@wp.kernel
def transform_points(
    src: wp.array(dtype=wp.vec3), dest: wp.array(dtype=wp.vec3), xform: wp.transform
):
    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(xform, p)

    dest[tid] = m


# Return interpenetration distances between query points (e.g., plug vertices in current pose)
# and mesh surfaces (e.g., of socket mesh in current pose)
@wp.kernel
def get_interpen_dist(
    queries: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    interpen_dists: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    # Declare arguments to wp.mesh_query_point() that will not be modified
    q = queries[tid]  # query point
    max_dist = 1.5  # max distance on mesh from query point

    # Declare arguments to wp.mesh_query_point() that will be modified
    sign = float(
        0.0
    )  # -1 if query point inside mesh; 0 if on mesh; +1 if outside mesh (NOTE: Mesh must be watertight!)
    face_idx = int(0)  # index of closest face
    face_u = float(0.0)  # barycentric u-coordinate of closest point
    face_v = float(0.0)  # barycentric v-coordinate of closest point

    # Get closest point on mesh to query point
    closest_mesh_point_exists = wp.mesh_query_point(
        mesh, q, max_dist, sign, face_idx, face_u, face_v
    )

    # If point exists within max_dist
    if closest_mesh_point_exists:
        # Get 3D position of point on mesh given face index and barycentric coordinates
        p = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)

        # Get signed distance between query point and mesh point
        delta = q - p
        signed_dist = sign * wp.length(delta)

        # If signed distance is negative
        if signed_dist < 0.0:
            # Store interpenetration distance
            interpen_dists[tid] = signed_dist
