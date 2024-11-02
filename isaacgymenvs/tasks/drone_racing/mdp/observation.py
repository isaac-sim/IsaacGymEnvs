from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch

from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import (
    quaternion_to_matrix,
    quat_rotate_inverse,
)
from ..waypoint import WaypointData


@dataclass
class ObservationParams:
    # number of parallel envs
    num_envs: int = 64

    # device to run tensor
    device: str = "cuda"

    # action dim
    dim_action: int = 4

    # drone position relative to start is clamped and normalized (m)
    pos_max: float = 40.0

    # vel of each channel is clamped at this value and normalized (m/s)
    vel_max: float = 20.0

    # angular vel of each channel is also clamped and normalized (rad/s)
    ang_vel_max: float = 6 * torch.pi

    # distances to waypoint corners are as well clamped and normalized (m)
    dist_to_corner_max: float = 25.0


class Observation:

    def __init__(self, params: ObservationParams):
        self.params = params
        self.all_env_id = torch.arange(params.num_envs, device=params.device)

        self.cam_tf_tensor = torch.zeros(params.num_envs, 12, device=params.device)

        self.init_drone_pos = torch.zeros(params.num_envs, 3, device=params.device)
        self.last_action = torch.zeros(
            params.num_envs, params.dim_action, device=params.device
        )

        self.wp_x_axis: Optional[torch.Tensor] = None
        self.wp_corner_pos: Optional[torch.Tensor] = None
        self.wp_center_pos: Optional[torch.Tensor] = None

    def set_waypoint_and_cam(
        self, wp_data: WaypointData, cam_tf: List[gymapi.Transform]
    ):
        # extract useful waypoint info
        num_waypoints = wp_data.num_waypoints
        wp_x_axis = torch.zeros(self.params.num_envs, num_waypoints, 3)
        wp_corner_pos = torch.zeros(self.params.num_envs, num_waypoints, 4, 3)

        for i in range(num_waypoints):
            # loop through waypoints, envs are still vectorized
            wp_q = wp_data.quaternion[:, i]
            wp_mat = quaternion_to_matrix(wp_q.roll(1, dims=1))
            wp_x_axis[:, i] = wp_mat[:, :, 0]  # (num_envs, 3)

            wp_p = wp_data.position[:, i]
            dw = wp_data.width[:, i].unsqueeze(1) / 2
            dh = wp_data.height[:, i].unsqueeze(1) / 2
            # upper left corner
            wp_corner_pos[:, i, 0] = wp_p + wp_mat[:, :, 1] * dw + wp_mat[:, :, 2] * dh
            # upper right corner
            wp_corner_pos[:, i, 1] = wp_p - wp_mat[:, :, 1] * dw + wp_mat[:, :, 2] * dh
            # lower right corner
            wp_corner_pos[:, i, 2] = wp_p - wp_mat[:, :, 1] * dw - wp_mat[:, :, 2] * dh
            # lower left corner
            wp_corner_pos[:, i, 3] = wp_p + wp_mat[:, :, 1] * dw - wp_mat[:, :, 2] * dh

        self.wp_x_axis = wp_x_axis.to(device=self.params.device)
        self.wp_corner_pos = wp_corner_pos.to(device=self.params.device)
        self.wp_center_pos = wp_data.position.to(device=self.params.device)

        # update cam_tf_tensor
        # TODO: optimize for large number of envs
        for i in range(self.params.num_envs):
            self.cam_tf_tensor[i, :3] = torch.tensor(
                [cam_tf[i].p.x, cam_tf[i].p.y, cam_tf[i].p.z], device=self.params.device
            )
            cam_q = torch.tensor(
                [cam_tf[i].r.x, cam_tf[i].r.y, cam_tf[i].r.z, cam_tf[i].r.w],
                device=self.params.device,
            )
            cam_mat = quaternion_to_matrix(cam_q.roll(1))
            self.cam_tf_tensor[i, 3:] = cam_mat.flatten()

        self.cam_tf_tensor.clamp_(min=-1.0, max=1.0)

    def set_init_drone_state_action(
        self,
        drone_state: torch.Tensor,
        action: torch.Tensor,
        env_id: torch.Tensor = None,
    ):
        if env_id is None:
            env_id = self.all_env_id

        self.init_drone_pos[env_id] = drone_state[env_id, :3]
        self.last_action[env_id] = action[env_id]

    def compute(
        self,
        drone_state: torch.Tensor,
        next_wp_id: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes observation terms. The waypoint info tensor is in (num_envs, 34), whose
        ``[:17]`` of last dim is the next waypoint's: [``cos_sim`` (1), ``dist_to_corners`` (4),
        ``unit_vec_to_corners`` (12)], and ``[17:]`` is the same info for the further next waypoint.

        Args:
            drone_state: full drone state tensor in (num_envs, 13).
            next_wp_id: 1-dim int tensor indicating next wp id in (num_envs, ).
            action: current action tensor in (num_envs, dim_action).

        Returns:
            - Flat drone state (normalized) [p, R, v, w] tensor in (num_envs, 18).
            - Camera pose (clamped -1 to 1) (p, R) in body frame tensor in (num_envs, 12).
            - Waypoint info (with dist normalized) tensor in (num_envs, 34).
            - Last action in (num_envs, dim_action).
        """

        flat_drone_state, wp_info, last_action = _compute_script(
            pos_max=self.params.pos_max,
            vel_max=self.params.vel_max,
            ang_vel_max=self.params.ang_vel_max,
            dist_to_corner_max=self.params.dist_to_corner_max,
            wp_x_axis=self.wp_x_axis,
            wp_corner_pos=self.wp_corner_pos,
            wp_center_pos=self.wp_center_pos,
            init_drone_p=self.init_drone_pos,
            last_action=self.last_action,
            all_env_id=self.all_env_id,
            drone_state=drone_state,
            next_wp_id=next_wp_id,
        )
        self.last_action[:] = action

        return flat_drone_state, self.cam_tf_tensor, wp_info, last_action


@torch.jit.script
def _compute_script(
    pos_max: float,
    vel_max: float,
    ang_vel_max: float,
    dist_to_corner_max: float,
    wp_x_axis: torch.Tensor,
    wp_corner_pos: torch.Tensor,
    wp_center_pos: torch.Tensor,
    init_drone_p: torch.Tensor,
    last_action: torch.Tensor,
    all_env_id: torch.Tensor,
    drone_state: torch.Tensor,
    next_wp_id: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # flat drone state tensor (num_envs, 18)
    drone_p = drone_state[:, :3]
    drone_q = drone_state[:, 3:7]
    drone_r_mat = quaternion_to_matrix(drone_q.roll(1, dims=1))
    drone_ang_vel_b = quat_rotate_inverse(drone_q, drone_state[:, 10:])
    flat_drone_state = torch.cat(
        (
            torch.clamp((drone_p - init_drone_p) / pos_max, min=-1, max=1),
            drone_r_mat.transpose(1, 2).flatten(1),  # better readability after flatten
            torch.clamp(drone_state[:, 7:10] / vel_max, min=-1, max=1),
            torch.clamp(drone_ang_vel_b / ang_vel_max, min=-1, max=1),
        ),
        dim=1,
    )

    # waypoint info tensor (num_envs, 34)
    wp_info = torch.zeros(wp_x_axis.shape[0], 34, device=wp_x_axis.device)
    max_wp_id = wp_x_axis.shape[1] - 1
    for i in range(2):
        # we need info of two future waypoints
        wp_id = next_wp_id + i
        wp_id.clamp_(max=max_wp_id)

        # cosine similarity between wp x-axis and vector to wp center
        next_wp_x_axis = wp_x_axis[all_env_id, wp_id]  # (num_envs, 3)
        next_wp_center = wp_center_pos[all_env_id, wp_id]
        vec_to_center = next_wp_center - drone_p
        cos_sim = torch.sum(next_wp_x_axis * vec_to_center, dim=-1) / (
            torch.linalg.norm(next_wp_x_axis, dim=-1)
            * torch.linalg.norm(vec_to_center, dim=-1)
        )
        cos_sim.nan_to_num_(nan=0.0)

        # relative corner position in drone body frame
        next_wp_corners = wp_corner_pos[all_env_id, wp_id]  # (num_envs, 4, 3)
        vec_to_corners_w = next_wp_corners - drone_p.unsqueeze(1)  # (num_envs, 4, 3)
        dist_to_corners = torch.linalg.norm(
            vec_to_corners_w, dim=-1, keepdim=True
        )  # (num_envs, 4, 1)
        q = drone_q.unsqueeze(1).expand(-1, 4, -1)
        vec_to_corners_b = quat_rotate_inverse(
            q.reshape(-1, 4), vec_to_corners_w.view(-1, 3)
        ).view(-1, 4, 3)
        unit_vec_b = vec_to_corners_b / dist_to_corners  # (num_envs, 4, 3)
        unit_vec_b.nan_to_num_(nan=0.0)

        # update wp_info tensor
        data_id_start = i * 17
        data_id_end = (i + 1) * 17
        wp_info[:, data_id_start:data_id_end] = torch.cat(
            (
                cos_sim.view(-1, 1),
                torch.clamp(
                    dist_to_corners.view(-1, 4) / dist_to_corner_max, min=0, max=1
                ),
                unit_vec_b.view(-1, 12),
            ),
            dim=-1,
        )

    return flat_drone_state, wp_info, last_action
