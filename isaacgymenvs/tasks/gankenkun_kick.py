# Copyright (c) 2018-2023, NVIDIA Corporation
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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class GankenKunKick(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        
        # normalization
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["ball_velocity"] = self.cfg["env"]["learn"]["ballVelocityRewardScale"]
        self.rew_scales["ball_distance"] = self.cfg["env"]["learn"]["ballDistanceRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["fall"] = self.cfg["env"]["learn"]["fallRewardScale"]
        self.rew_scales["goal"] = self.cfg["env"]["learn"]["goalRewardScale"]
        self.rew_scales["ball_out"] = self.cfg["env"]["learn"]["ballOutRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # ball init state
        ball_pos = self.cfg["env"]["ballInitState"]["pos"]
        ball_rot = self.cfg["env"]["ballInitState"]["rot"]
        ball_v_lin = self.cfg["env"]["ballInitState"]["vLinear"]
        ball_v_ang = self.cfg["env"]["ballInitState"]["vAngular"]
        ball_state = ball_pos + ball_rot + ball_v_lin + ball_v_ang

        field_state = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.root_init_state = [state, ball_state, field_state]

        # action_list
        fwd, rev = 1, -1
        self.named_action_list = [
            [["right_ankle_roll_joint", fwd]],
            [["right_ankle_pitch_joint", fwd]],
            [["right_knee_pitch_joint", fwd], ["right_ankle_pitch_mimic_joint", rev],["right_shin_pitch_mimic_joint", fwd]],
            [["right_waist_pitch_joint", fwd], ["right_knee_pitch_mimic_joint", rev],["right_waist_pitch_mimic_joint", fwd]],
            [["right_waist_roll_joint", fwd]],
            [["right_waist_yaw_joint", fwd]],
            [["right_shoulder_pitch_joint", fwd]],
            [["right_shoulder_roll_joint", fwd]],
            [["right_elbow_pitch_joint", fwd]],
            [["left_ankle_roll_joint", fwd]],
            [["left_ankle_pitch_joint", fwd]],
            [["left_knee_pitch_joint", fwd], ["left_ankle_pitch_mimic_joint", rev],["left_shin_pitch_mimic_joint", fwd]],
            [["left_waist_pitch_joint", fwd], ["left_knee_pitch_mimic_joint", rev],["left_waist_pitch_mimic_joint", fwd]],
            [["left_waist_roll_joint", fwd]],
            [["left_waist_yaw_joint", fwd]],
            [["left_shoulder_pitch_joint", fwd]],
            [["left_shoulder_roll_joint", fwd]],
            [["left_elbow_pitch_joint", fwd]]
        ]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 32
        self.cfg["env"]["numActions"] = 19

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # action to joint id and direction
        self.action_list = [[] for _ in range(len(self.named_action_list))]
        for i in range(len(self.named_action_list)):
            actions = self.named_action_list[i]
            for action in actions:
                self.action_list[i].append([self.dof_names.index(action[0]),action[1]])

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.walking_period = self.cfg["env"]["control"]["walkingPeriod"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        actors_per_env = 3
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, actors_per_env, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(len(self.dof_pos[0])):
            name = self.dof_names[i]
            if name in self.named_default_joint_angles.keys():
                angle = self.named_default_joint_angles[name]
                self.default_dof_pos[:, i] = angle
        for actions in self.action_list:
            if len(actions) > 1:
                for i in range(1, len(actions)):
                    self.default_dof_pos[:, actions[i][0]] = actions[i][1] * self.default_dof_pos[:, actions[0][0]]

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.root_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.forward_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.goal_pos = to_torch([4.5, 0], device=self.device).repeat((self.num_envs, 1))
        self.walking_phase = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"ball_velocity": torch_zeros(), "ball_distance": torch_zeros(), "torque": torch_zeros(), "fall": torch_zeros(), "goal": torch_zeros(), "ball_out": torch_zeros()}

        self.all_actor_indices = torch.arange(actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, actors_per_env)
        self.all_gankenkun_indices = actors_per_env * torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/GankenKun/gankenkun.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.007
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        gankenkun_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(gankenkun_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(gankenkun_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(gankenkun_asset)
        self.dof_names = self.gym.get_asset_dof_names(gankenkun_asset)
        feet_names = [s for s in body_names if "ankle_roll_link" in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "knee_pitch_link" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(gankenkun_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            #dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("effort: "+str(dof_props['effort']))
        print("velocity: "+str(dof_props['velocity']))
        print("friction: "+str(dof_props['friction']))
        print("hasLimits: "+str(dof_props['hasLimits']))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # create ball asset
        ball_asset_file = "urdf/GankenKun/ball.urdf"
        ball_options = gymapi.AssetOptions()
        ball_options.angular_damping = 0.77
        ball_options.linear_damping = 0.77
        ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, ball_options)
        ball_init_pose = gymapi.Transform()
        ball_init_pose.p = gymapi.Vec3(0.2, 0, 0.07)

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.gankenkun_handles = []
        self.ball_handles = []
        self.field_handles = []
        self.envs = []

        # create field asset
        field_asset_file = "urdf/GankenKun/field.urdf"
        field_options = gymapi.AssetOptions()
        field_options.flip_visual_attachments = True
        field_options.fix_base_link = True
        field_options.collapse_fixed_joints = True
        field_options.disable_gravity = True
        field_options.thickness = 0.001
        field_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        field_options.use_mesh_materials = True
        field_asset = self.gym.load_asset(self.sim, asset_root, field_asset_file, field_options)
        field_init_pose = gymapi.Transform()
        field_init_pose.p = gymapi.Vec3(0.4, 0, 0.07)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            gankenkun_handle = self.gym.create_actor(env_ptr, gankenkun_asset, start_pose, "gankenkun", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, gankenkun_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, gankenkun_handle)
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_init_pose, "ball", i, 1, 0)
            self.ball_handles.append(ball_handle)
            field_handle = self.gym.create_actor(env_ptr, field_asset, field_init_pose, "field", i, 2, 0)
            self.field_handles.append(field_handle)
            self.envs.append(env_ptr)
            self.gankenkun_handles.append(gankenkun_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.gankenkun_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.gankenkun_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.gankenkun_handles[0], "base_link")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
#        self.actions[:,:] = 0

        self.walking_phase = (self.walking_phase+self.dt/self.walking_period)%1.0
        self.walking_phase[:,0] += self.actions[:,18]*self.dt/self.walking_period*0.5

        right = 1-abs(2*((self.walking_phase*2)%1)*(self.walking_phase%1<0.5)-1)
        left = 1-abs(2*((self.walking_phase*2)%1)*(self.walking_phase%1>0.5)-1)
        self.actions[:,2] += right[:,0]
        self.actions[:,3] -= right[:,0]
        self.actions[:,11] += left[:,0]
        self.actions[:,12] -= left[:,0]

        for i in range(len(self.action_list)):
            actions = self.action_list[i]
            for j in range(len(actions)):
                self.joint_actions[:, actions[j][0]] = actions[j][1] * self.actions[:, i]

        targets = self.action_scale * self.joint_actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], rew_ball_velocity, rew_ball_distance, rew_torque, rew_fall, rew_goal, rew_ball_out = compute_gankenkun_reward(
            # tensors
            self.root_states,
            self.torques,
            self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            self.goal_pos,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
        )
        # log episode reward sums
        self.episode_sums["ball_velocity"] += rew_ball_velocity
        self.episode_sums["ball_distance"] += rew_ball_distance
        self.episode_sums["torque"] += rew_torque
        self.episode_sums["fall"] += rew_fall
        self.episode_sums["goal"] += rew_goal
        self.episode_sums["ball_out"] += rew_ball_out

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_gankenkun_observations(  # tensors
                                                        self.root_states,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.forward_vec,
                                                        self.actions,
                                                        self.walking_phase,
                                                        # scales
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        actor_indices = self.all_actor_indices[env_ids_int32].flatten()

        #ball_pos = torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        self.root_states[env_ids] = to_torch(self.root_init_state, device=self.device, requires_grad=False)

        self.root_states[env_ids, 1, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

        gankenkun_indices = self.all_gankenkun_indices[env_ids].flatten()

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(gankenkun_indices), len(gankenkun_indices))

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / self.progress_buf[env_ids])
            self.episode_sums[key][env_ids] = 0.

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_gankenkun_reward(
    # tensors
    root_states,
    torques,
    contact_forces,
    knee_indices,
    episode_lengths,
    goal_pos,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    is_fall = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    is_fall = is_fall | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)

    ball_velocity = root_states[:, 1, 7]
    rew_ball_velocity = torch.clip(ball_velocity, 0, None) * rew_scales["ball_velocity"]

    ball_pos = root_states[:, 1, :2]
    base_pos = root_states[:, 0, :2]
    ball_distance = torch.sum(torch.square(ball_pos - base_pos), dim=1)
    rew_ball_distance = torch.exp(-ball_distance/0.25) * rew_scales["ball_distance"]

    # torque penalty
    rew_torque = torch.sum(torch.abs(torques), dim=1) * rew_scales["torque"]

    rew_fall = is_fall * rew_scales["fall"]

    # goal

    ball_out = ball_pos[:, 0] > 4.5
    is_goal = ball_out & (torch.abs(ball_pos[:, 1]) < 1.3)
    ball_out = ball_out | (torch.abs(ball_pos[:, 1]) > 3.0)

    rew_goal = is_goal * rew_scales["goal"]
    rew_ball_out = ball_out * rew_scales["ball_out"]

    total_reward = rew_ball_velocity + rew_ball_distance + rew_torque + rew_fall + rew_goal + rew_ball_out

    # reset agents
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = is_fall | time_out | ball_out

    return total_reward.detach(), reset, rew_ball_velocity, rew_ball_distance, rew_torque, rew_fall, rew_goal, rew_ball_out

@torch.jit.script
def compute_gankenkun_observations(root_states,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                forward_vec,
                                actions,
                                walking_phase,
                                dof_pos_scale,
                                dof_vel_scale
                                ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    base_quat = root_states[:, 0, 3:7]
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale
    projected_forward = quat_rotate(base_quat, forward_vec)
    base_pos = root_states[:, 0, 0:3]
    ball_pos = root_states[:, 1, 0:3]
    local_ball_pos = quat_rotate_inverse(base_quat, ball_pos - base_pos)

    obs = torch.cat((projected_gravity,
                     actions,
                     base_pos,
                     local_ball_pos,
                     projected_forward,
                     walking_phase
                     ), dim=-1)

    return obs
