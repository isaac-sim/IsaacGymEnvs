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

import math
import os
import tempfile
from copy import copy
from os.path import join
from typing import List, Tuple

from isaacgym import gymapi, gymtorch, gymutil
from torch import Tensor

from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import DofParameters, populate_dof_properties
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.allegro_kuka.generate_cuboids import (
    generate_big_cuboids,
    generate_default_cube,
    generate_small_cuboids,
    generate_sticks,
)
from isaacgymenvs.utils.torch_jit_utils import *


class AllegroKukaTwoArmsBase(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.frame_since_restart: int = 0  # number of control steps since last restart across all actors

        self.hand_arm_asset_file: str = self.cfg["env"]["asset"]["kukaAllegro"]

        self.clamp_abs_observations: float = self.cfg["env"]["clampAbsObservations"]

        self.num_arms = self.cfg["env"]["numArms"]
        assert self.num_arms == 2, f"Only two arms supported, got {self.num_arms}"

        self.arm_x_ofs = self.cfg["env"]["armXOfs"]
        self.arm_y_ofs = self.cfg["env"]["armYOfs"]

        # 4 joints for index, middle, ring, and thumb and 7 for kuka arm
        self.num_arm_dofs = 7
        self.num_finger_dofs = 4
        self.num_allegro_fingertips = 4
        self.num_hand_dofs = self.num_finger_dofs * self.num_allegro_fingertips
        self.num_hand_arm_dofs = self.num_hand_dofs + self.num_arm_dofs

        self.num_allegro_kuka_actions = self.num_hand_arm_dofs * self.num_arms

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]

        # not used in 2-arm task for now
        # to fix: add to config
        # self.kuka_actions_penalty_scale = self.cfg["env"]["kukaActionsPenaltyScale"]
        # self.allegro_actions_penalty_scale = self.cfg["env"]["allegroActionsPenaltyScale"]

        self.dof_params: DofParameters = DofParameters.from_cfg(self.cfg)

        self.initial_tolerance = self.cfg["env"]["successTolerance"]
        self.success_tolerance = self.initial_tolerance
        self.target_tolerance = self.cfg["env"]["targetSuccessTolerance"]
        self.tolerance_curriculum_increment = self.cfg["env"]["toleranceCurriculumIncrement"]
        self.tolerance_curriculum_interval = self.cfg["env"]["toleranceCurriculumInterval"]

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

        self.reset_position_noise_x = self.cfg["env"]["resetPositionNoiseX"]
        self.reset_position_noise_y = self.cfg["env"]["resetPositionNoiseY"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise_fingers = self.cfg["env"]["resetDofPosRandomIntervalFingers"]
        self.reset_dof_pos_noise_arm = self.cfg["env"]["resetDofPosRandomIntervalArm"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        # currently not used in 2-hand env
        # self.hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.success_steps: int = self.cfg["env"]["successSteps"]

        # 1.0 means keypoints correspond to the corners of the object
        # larger values help the agent to prioritize rotation matching
        self.keypoint_scale = self.cfg["env"]["keypointScale"]

        # size of the object (i.e. cube) before scaling
        self.object_base_size = self.cfg["env"]["objectBaseSize"]

        # whether to sample random object dimensions
        self.randomize_object_dimensions = self.cfg["env"]["randomizeObjectDimensions"]
        self.with_small_cuboids = self.cfg["env"]["withSmallCuboids"]
        self.with_big_cuboids = self.cfg["env"]["withBigCuboids"]
        self.with_sticks = self.cfg["env"]["withSticks"]

        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (self.control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",  # 0.05m box
            "table": "urdf/table_wide.urdf",
            "bucket": "urdf/objects/bucket.urdf",
            "lightbulb": "lightbulb/A60_E27_SI.urdf",
            "socket": "E27SocketSimple.urdf",
            "ball": "urdf/objects/ball.urdf",
        }

        self.keypoints_offsets = self._object_keypoint_offsets()

        self.num_keypoints = len(self.keypoints_offsets)

        self.allegro_fingertips = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        self.fingertip_offsets = np.array(
            [[0.05, 0.005, 0], [0.05, 0.005, 0], [0.05, 0.005, 0], [0.06, 0.005, 0]], dtype=np.float32
        )
        palm_offset = np.array([-0.00, -0.02, 0.16], dtype=np.float32)

        self.num_fingertips = len(self.allegro_fingertips)

        # can be only "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_state"]):
            raise Exception("Unknown type of observations!")

        print("Obs type:", self.obs_type)

        num_dof_pos = num_dof_vel = self.num_hand_arm_dofs * self.num_arms

        palm_pos_size = 3 * self.num_arms
        palm_rot_vel_angvel_size = 10 * self.num_arms

        obj_rot_vel_angvel_size = 10

        fingertip_rel_pos_size = 3 * self.num_fingertips * self.num_arms

        keypoints_rel_palm_size = self.num_keypoints * 3 * self.num_arms
        keypoints_rel_goal_size = self.num_keypoints * 3

        object_scales_size = 3
        max_keypoint_dist_size = 1
        lifted_object_flag_size = 1
        progress_obs_size = 1 + 1

        # commented out for now - not used in 2-hand env
        # closest_fingertip_distance_size = self.num_fingertips * self.num_arms
        reward_obs_size = 1

        self.full_state_size = (
            num_dof_pos
            + num_dof_vel
            + palm_pos_size
            + palm_rot_vel_angvel_size
            + obj_rot_vel_angvel_size
            + fingertip_rel_pos_size
            + keypoints_rel_palm_size
            + keypoints_rel_goal_size
            + object_scales_size
            + max_keypoint_dist_size
            + lifted_object_flag_size
            + progress_obs_size
            + reward_obs_size
        )

        num_states = self.full_state_size

        self.num_obs_dict = {
            "full_state": self.full_state_size,
        }

        self.up_axis = "z"

        self.fingertip_obs = True

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = self.num_allegro_kuka_actions

        self.cfg["device_type"] = sim_device.split(":")[0]
        self.cfg["device_id"] = int(sim_device.split(":")[1])
        self.cfg["headless"] = headless

        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        if self.viewer is not None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # volume to sample target position from
        target_volume_origin = np.array([0, 0.0, 0.8], dtype=np.float32)
        target_volume_extent = np.array([[-0.2, 0.2], [-0.5, 0.5], [-0.12, 0.25]], dtype=np.float32)

        self.target_volume_origin = torch.from_numpy(target_volume_origin).to(self.device).float()
        self.target_volume_extent = torch.from_numpy(target_volume_extent).to(self.device).float()

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.hand_arm_default_dof_pos = torch.zeros(
            [self.num_arms, self.num_hand_arm_dofs], dtype=torch.float, device=self.device
        )

        desired_kuka_pos = torch.tensor([-1.571, 1.571, -0.000, 1.6, -0.000, 1.485, 2.358])  # pose v1
        # desired_kuka_pos = torch.tensor([-2.135, 0.843, 1.786, -0.903, -2.262, 1.301, -2.791])  # pose v2
        self.hand_arm_default_dof_pos[0, :7] = desired_kuka_pos

        desired_kuka_pos = torch.tensor([-1.571, 1.571, -0.000, 1.6, -0.000, 1.485, 2.358])  # pose v1
        # desired_kuka_pos = torch.tensor([-2.135, 0.843, 1.786, -0.903, -2.262, 1.301, -2.791])  # pose v2
        self.hand_arm_default_dof_pos[1, :7] = desired_kuka_pos

        self.pos_noise_coeff = torch.zeros_like(self.hand_arm_default_dof_pos, device=self.device)
        self.pos_noise_coeff[:, 0:7] = self.reset_dof_pos_noise_arm
        self.pos_noise_coeff[:, 7 : self.num_hand_arm_dofs] = self.reset_dof_pos_noise_fingers
        self.pos_noise_coeff = self.pos_noise_coeff.flatten()
        self.hand_arm_default_dof_pos = self.hand_arm_default_dof_pos.flatten()

        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_hand_arm_dofs * self.num_arms]
        # this will have dimensions [num_envs, num_arms * num_hand_arm_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.palm_center_offset = torch.from_numpy(palm_offset).to(self.device).repeat((self.num_envs, 1))
        self.palm_center_pos = torch.zeros((self.num_envs, self.num_arms, 3), dtype=torch.float, device=self.device)

        self.fingertip_offsets = torch.from_numpy(self.fingertip_offsets).to(self.device).repeat((self.num_envs, 1, 1))

        self.set_actor_root_state_object_indices: List[Tensor] = []

        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_arms * self.num_hand_arm_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_arms * self.num_hand_arm_dofs), dtype=torch.float, device=self.device
        )

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)

        # true objective value for the whole episode, plus saving values for the previous episode
        self.true_objective = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_true_objective = torch.zeros_like(self.true_objective)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.action_torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )
        self.goal_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )

        # how many steps we were within the goal tolerance
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        self.closest_fingertip_dist = -torch.ones(
            [self.num_envs, self.num_arms, self.num_fingertips], dtype=torch.float, device=self.device
        )

        reward_keys = [
            "raw_fingertip_delta_rew",
            "raw_lifting_rew",
            "raw_keypoint_rew",
            "fingertip_delta_rew",
            "lifting_rew",
            "lift_bonus_rew",
            "keypoint_rew",
            "bonus_rew",
        ]

        self.rewards_episode = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_keys
        }

        self.last_curriculum_update = 0

        self.episode_root_state_tensors = [[] for _ in range(self.num_envs)]
        self.episode_dof_states = [[] for _ in range(self.num_envs)]

        self.eval_stats: bool = self.cfg["env"]["evalStats"]
        if self.eval_stats:
            self.last_success_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.success_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.total_num_resets = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.successes_count = torch.zeros(
                self.max_consecutive_successes + 1, dtype=torch.float, device=self.device
            )
            from tensorboardX import SummaryWriter

            self.eval_summary_dir = "./eval_summaries"
            # remove the old directory if it exists
            if os.path.exists(self.eval_summary_dir):
                import shutil

                shutil.rmtree(self.eval_summary_dir)
            self.eval_summaries = SummaryWriter(self.eval_summary_dir, flush_secs=3)

    # AllegroKukaBase abstract interface - to be overriden in derived classes

    def _object_keypoint_offsets(self):
        raise NotImplementedError()

    def _object_start_pose(self, arms_y_ofs: float, table_pose_dy: float, table_pose_dz: float):
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = 0.0

        pose_dy, pose_dz = table_pose_dy, table_pose_dz + 0.25

        object_start_pose.p.y = arms_y_ofs + pose_dy
        object_start_pose.p.z = pose_dz

        return object_start_pose

    def _main_object_assets_and_scales(self, object_asset_root, tmp_assets_dir):
        object_asset_files, object_asset_scales = self._box_asset_files_and_scales(object_asset_root, tmp_assets_dir)
        if not self.randomize_object_dimensions:
            object_asset_files = object_asset_files[:1]
            object_asset_scales = object_asset_scales[:1]

        # randomize order
        files_and_scales = list(zip(object_asset_files, object_asset_scales))

        # use fixed seed here to make sure when we restart from checkpoint the distribution of object types is the same
        rng = np.random.default_rng(42)
        rng.shuffle(files_and_scales)

        object_asset_files, object_asset_scales = zip(*files_and_scales)
        return object_asset_files, object_asset_scales

    def _load_main_object_asset(self):
        """Load manipulated object and goal assets."""
        object_asset_options = gymapi.AssetOptions()
        object_assets = []
        for object_asset_file in self.object_asset_files:
            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)
            object_asset_ = self.gym.load_asset(self.sim, object_asset_dir, object_asset_fname, object_asset_options)
            object_assets.append(object_asset_)
        object_rb_count = self.gym.get_asset_rigid_body_count(
            object_assets[0]
        )  # assuming all of them have the same rb count
        object_shapes_count = self.gym.get_asset_rigid_shape_count(
            object_assets[0]
        )  # assuming all of them have the same rb count
        return object_assets, object_rb_count, object_shapes_count

    def _load_additional_assets(self, object_asset_root, arm_y_offset: float) -> Tuple[int, int]:
        """
        returns: tuple (num_rigid_bodies, num_shapes)
        """
        return 0, 0

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        pass

    def _after_envs_created(self):
        pass

    def _extra_reset_rules(self, resets):
        return resets

    def _reset_target(self, env_ids: Tensor) -> None:
        raise NotImplementedError()

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        return []

    def _extra_curriculum(self):
        pass

    # AllegroKukaBase implementation
    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return dict(
            success_tolerance=self.success_tolerance,
        )

    def set_env_state(self, env_state):
        if env_state is None:
            return

        for key in self.get_env_state().keys():
            value = env_state.get(key, None)
            if value is None:
                continue

            self.__dict__[key] = value
            print(f"Loaded env state value {key}:{value}")

        print(f"Success tolerance value after loading from checkpoint: {self.success_tolerance}")

    # noinspection PyMethodOverriding
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2 (same as in allegro_hand.py)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _box_asset_files_and_scales(self, object_assets_root, generated_assets_dir):
        files = []
        scales = []

        try:
            filenames = os.listdir(generated_assets_dir)
            for fname in filenames:
                if fname.endswith(".urdf"):
                    os.remove(join(generated_assets_dir, fname))
        except Exception as exc:
            print(f"Exception {exc} while removing older procedurally-generated urdf assets")

        objects_rel_path = os.path.dirname(self.asset_files_dict[self.object_type])
        objects_dir = join(object_assets_root, objects_rel_path)
        base_mesh = join(objects_dir, "meshes", "cube_multicolor.obj")

        generate_default_cube(generated_assets_dir, base_mesh, self.object_base_size)

        if self.with_small_cuboids:
            generate_small_cuboids(generated_assets_dir, base_mesh, self.object_base_size)
        if self.with_big_cuboids:
            generate_big_cuboids(generated_assets_dir, base_mesh, self.object_base_size)
        if self.with_sticks:
            generate_sticks(generated_assets_dir, base_mesh, self.object_base_size)

        filenames = os.listdir(generated_assets_dir)
        filenames = sorted(filenames)

        for fname in filenames:
            if fname.endswith(".urdf"):
                scale_tokens = os.path.splitext(fname)[0].split("_")[2:]
                files.append(join(generated_assets_dir, fname))
                scales.append([float(scale_token) / 100 for scale_token in scale_tokens])

        return files, scales

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")

        object_asset_root = asset_root
        tmp_assets_dir = tempfile.TemporaryDirectory()
        self.object_asset_files, self.object_asset_scales = self._main_object_assets_and_scales(
            object_asset_root, tmp_assets_dir.name
        )

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {self.hand_arm_asset_file} from {asset_root}")
        allegro_kuka_asset = self.gym.load_asset(self.sim, asset_root, self.hand_arm_asset_file, asset_options)
        print(f"Loaded asset {allegro_kuka_asset}")

        num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(allegro_kuka_asset)
        num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(allegro_kuka_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(allegro_kuka_asset)
        assert (
            self.num_hand_arm_dofs == num_hand_arm_dofs
        ), f"Number of DOFs in asset {allegro_kuka_asset} is {num_hand_arm_dofs}, but {self.num_hand_arm_dofs} was expected"

        max_agg_bodies = all_arms_bodies = num_hand_arm_bodies * self.num_arms
        max_agg_shapes = all_arms_shapes = num_hand_arm_shapes * self.num_arms

        allegro_rigid_body_names = [
            self.gym.get_asset_rigid_body_name(allegro_kuka_asset, i) for i in range(num_hand_arm_bodies)
        ]
        print(f"Allegro num rigid bodies: {num_hand_arm_bodies}")
        print(f"Allegro rigid bodies: {allegro_rigid_body_names}")

        # allegro_actuated_dof_names = [self.gym.get_asset_actuator_joint_name(allegro_asset, i) for i in range(self.num_allegro_dofs)]
        # self.allegro_actuated_dof_indices = [self.gym.find_asset_dof_index(allegro_asset, name) for name in allegro_actuated_dof_names]

        hand_arm_dof_props = self.gym.get_asset_dof_properties(allegro_kuka_asset)

        arm_hand_dof_lower_limits = []
        arm_hand_dof_upper_limits = []

        for arm_idx in range(self.num_arms):
            for i in range(self.num_hand_arm_dofs):
                arm_hand_dof_lower_limits.append(hand_arm_dof_props["lower"][i])
                arm_hand_dof_upper_limits.append(hand_arm_dof_props["upper"][i])

        # self.allegro_actuated_dof_indices = to_torch(self.allegro_actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(arm_hand_dof_upper_limits, device=self.device)

        arm_poses = [gymapi.Transform() for _ in range(self.num_arms)]
        arm_x_ofs, arm_y_ofs = self.arm_x_ofs, self.arm_y_ofs
        for arm_idx, arm_pose in enumerate(arm_poses):
            x_ofs = arm_x_ofs * (-1 if arm_idx == 0 else 1)
            arm_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(x_ofs, arm_y_ofs, 0)

            # arm_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            if arm_idx == 0:
                # rotate 1st arm 90 degrees to the left
                arm_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi / 2)
            else:
                # rotate 2nd arm 90 degrees to the right
                arm_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi / 2)

        object_assets, object_rb_count, object_shapes_count = self._load_main_object_asset()
        max_agg_bodies += object_rb_count
        max_agg_shapes += object_shapes_count

        # load auxiliary objects
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = False
        table_asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(self.sim, asset_root, self.asset_files_dict["table"], table_asset_options)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = 0.0
        # table_pose_dy, table_pose_dz = -0.8, 0.38
        table_pose_dy, table_pose_dz = 0.0, 0.38
        table_pose.p.y = arm_y_ofs + table_pose_dy
        table_pose.p.z = table_pose_dz

        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_bodies += table_rb_count
        max_agg_shapes += table_shapes_count

        additional_rb, additional_shapes = self._load_additional_assets(object_asset_root, arm_y_ofs)
        max_agg_bodies += additional_rb
        max_agg_shapes += additional_shapes

        # set up object and goal positions
        self.object_start_pose = self._object_start_pose(arm_y_ofs, table_pose_dy, table_pose_dz)

        self.envs = []

        object_init_state = []
        object_scales = []
        object_keypoint_offsets = []

        allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "iiwa7_link_7")
        fingertip_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.allegro_fingertips
        ]

        self.allegro_palm_handles = []
        self.allegro_fingertip_handles = []
        for arm_idx in range(self.num_arms):
            self.allegro_palm_handles.append(allegro_palm_handle + arm_idx * num_hand_arm_bodies)
            self.allegro_fingertip_handles.extend([h + arm_idx * num_hand_arm_bodies for h in fingertip_handles])

        # does this rely on the fact that objects are added right after the arms in terms of create_actor()?
        self.object_rb_handles = list(range(all_arms_bodies, all_arms_bodies + object_rb_count))

        self.arm_indices = torch.empty([self.num_envs, self.num_arms], dtype=torch.long, device=self.device)
        self.object_indices = torch.empty(self.num_envs, dtype=torch.long, device=self.device)

        assert self.num_envs >= 1
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add arms
            for arm_idx in range(self.num_arms):
                arm = self.gym.create_actor(env_ptr, allegro_kuka_asset, arm_poses[arm_idx], f"arm{arm_idx}", i, -1, 0)

                populate_dof_properties(hand_arm_dof_props, self.dof_params, self.num_arm_dofs, self.num_hand_dofs)

                self.gym.set_actor_dof_properties(env_ptr, arm, hand_arm_dof_props)
                allegro_hand_idx = self.gym.get_actor_index(env_ptr, arm, gymapi.DOMAIN_SIM)

                self.arm_indices[i, arm_idx] = allegro_hand_idx

            # add object
            object_asset_idx = i % len(object_assets)
            object_asset = object_assets[object_asset_idx]

            obj_pose = self.object_start_pose
            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, "object", i, 0, 0)
            pos, rot = obj_pose.p, obj_pose.r
            object_init_state.append([pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w, 0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices[i] = object_idx

            object_scale = self.object_asset_scales[object_asset_idx]
            object_scales.append(object_scale)
            object_offsets = []
            for keypoint in self.keypoints_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= object_scale[coord_idx] * self.object_base_size * self.keypoint_scale / 2
                object_offsets.append(keypoint)

            object_keypoint_offsets.append(object_offsets)

            # table object
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table_object", i, 0, 0)
            _table_object_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)

            # task-specific objects (i.e. goal object for reorientation task)
            self._create_additional_objects(env_ptr, env_idx=i, object_asset_idx=object_asset_idx)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        # noinspection PyUnboundLocalVariable
        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.allegro_fingertip_handles = to_torch(self.allegro_fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.object_scales = to_torch(object_scales, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets = to_torch(object_keypoint_offsets, dtype=torch.float, device=self.device)

        self._after_envs_created()

        try:
            # by this point we don't need the temporary folder for procedurally generated assets
            tmp_assets_dir.cleanup()
        except Exception:
            pass

    def _distance_delta_rewards(self, lifted_object: Tensor) -> Tensor:
        """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""
        # this is positive if we got closer, negative if we're further away than the closest we've gotten
        fingertip_deltas_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        # update the values if finger tips got closer to the object
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)

        # clip between zero and +inf to turn deltas into rewards
        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        fingertip_delta_rew = torch.sum(fingertip_delta_rew, dim=-1)  # sum over all arms

        # vvvv this is commented out for 2 arms: we want the 2nd arm to be relatively close at all times

        # add this reward only before the object is lifted off the table
        # after this, we should be guided only by keypoint and bonus rewards
        # fingertip_delta_rew *= ~lifted_object

        return fingertip_delta_rew

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""

        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)

        # this flag tells us if we lifted an object above a certain height compared to the initial position
        lifted_object = (z_lift > self.lifting_bonus_threshold) | self.lifted_object

        # Since we stop rewarding the agent for height after the object is lifted, we should give it large positive reward
        # to compensate for "lost" opportunity to get more lifting reward for sitting just below the threshold.
        # This bonus depends on the max lifting reward (lifting reward coeff * threshold) and the discount factor
        # (i.e. the effective future horizon for the agent)
        # For threshold 0.15, lifting reward coeff = 3 and gamma 0.995 (effective horizon ~500 steps)
        # a value of 300 for the bonus reward seems reasonable
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted_above_threshold

        # stop giving lifting reward once we crossed the threshold - now the agent can focus entirely on the
        # keypoint reward
        lifting_rew *= ~lifted_object

        # update the flag that describes whether we lifted an object above the table or not
        self.lifted_object = lifted_object
        return lifting_rew, lift_bonus_rew, lifted_object

    def _keypoint_reward(self, lifted_object: Tensor) -> Tensor:
        # this is positive if we got closer, negative if we're further away
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist

        # update the values if we got closer to the target
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)

        # clip between zero and +inf to turn deltas into rewards
        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)

        # administer reward only when we already lifted an object from the table
        # to prevent the situation where the agent just rolls it around the table
        keypoint_rew = max_keypoint_deltas * lifted_object

        return keypoint_rew

    def _compute_resets(self, is_success):
        resets = torch.where(self.object_pos[:, 2] < 0.1, torch.ones_like(self.reset_buf), self.reset_buf)  # fall
        if self.max_consecutive_successes > 0:
            # Reset progress buffer if max_consecutive_successes > 0
            self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
            resets = torch.where(self.successes >= self.max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(resets), resets)
        resets = self._extra_reset_rules(resets)
        return resets

    def _true_objective(self):
        raise NotImplementedError()

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew = self._distance_delta_rewards(lifted_object)
        keypoint_rew = self._keypoint_reward(lifted_object)

        keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale

        # noinspection PyTypeChecker
        near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance
        self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= self.success_steps
        goal_resets = is_success
        self.successes += is_success

        self.reset_goal_buf[:] = goal_resets

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew

        fingertip_delta_rew *= self.distance_delta_rew_scale
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        # We spread out the reward over "success_steps"
        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)

        reward = fingertip_delta_rew + lifting_rew + lift_bonus_rew + keypoint_rew + bonus_rew

        self.rew_buf[:] = reward

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        self.extras["successes"] = self.prev_episode_successes.mean()
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        # scalars for logging
        self.extras["true_objective_mean"] = self.true_objective.mean()
        self.extras["true_objective_min"] = self.true_objective.min()
        self.extras["true_objective_max"] = self.true_objective.max()

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (bonus_rew, "bonus_rew"),
        ]

        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        return self.rew_buf, is_success

    def _eval_stats(self, is_success: Tensor) -> None:
        if self.eval_stats:
            frame: int = self.frame_since_restart
            n_frames = torch.empty_like(self.last_success_step).fill_(frame)
            self.success_time = torch.where(is_success, n_frames - self.last_success_step, self.success_time)
            self.last_success_step = torch.where(is_success, n_frames, self.last_success_step)
            mask_ = self.success_time > 0
            if any(mask_):
                avg_time_mean = ((self.success_time * mask_).sum(dim=0) / mask_.sum(dim=0)).item()
            else:
                avg_time_mean = math.nan

            self.total_resets = self.total_resets + self.reset_buf.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            self.total_num_resets += self.reset_buf

            reset_ids = self.reset_buf.nonzero().squeeze()
            last_successes = self.successes[reset_ids].long()
            self.successes_count[last_successes] += 1

            if frame % 100 == 0:
                # The direct average shows the overall result more quickly, but slightly undershoots long term
                # policy performance.
                print(f"Max num successes: {self.successes.max().item()}")
                print(f"Average consecutive successes: {self.prev_episode_successes.mean().item():.2f}")
                print(f"Total num resets: {self.total_num_resets.sum().item()} --> {self.total_num_resets}")
                print(f"Reset percentage: {(self.total_num_resets > 0).sum() / self.num_envs:.2%}")
                print(f"Last ep successes: {self.prev_episode_successes.mean().item():.2f}")
                print(f"Last ep true objective: {self.prev_episode_true_objective.mean().item():.2f}")

                self.eval_summaries.add_scalar("last_ep_successes", self.prev_episode_successes.mean().item(), frame)
                self.eval_summaries.add_scalar(
                    "last_ep_true_objective", self.prev_episode_true_objective.mean().item(), frame
                )
                self.eval_summaries.add_scalar(
                    "reset_stats/reset_percentage", (self.total_num_resets > 0).sum() / self.num_envs, frame
                )
                self.eval_summaries.add_scalar("reset_stats/min_num_resets", self.total_num_resets.min().item(), frame)

                self.eval_summaries.add_scalar("policy_speed/avg_success_time_frames", avg_time_mean, frame)
                frame_time = self.control_freq_inv * self.dt
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_time_seconds", avg_time_mean * frame_time, frame
                )
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_per_minute", 60.0 / (avg_time_mean * frame_time), frame
                )
                print(f"Policy speed (successes per minute): {60.0 / (avg_time_mean * frame_time):.2f}")

                # create a matplotlib bar chart of the self.successes_count
                import matplotlib.pyplot as plt

                plt.bar(list(range(self.max_consecutive_successes + 1)), self.successes_count.cpu().numpy())
                plt.title("Successes histogram")
                plt.xlabel("Successes")
                plt.ylabel("Frequency")
                plt.savefig(f"{self.eval_summary_dir}/successes_histogram.png")
                plt.clf()

    def compute_observations(self) -> Tuple[Tensor, int]:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self._palm_state = self.rigid_body_states[:, self.allegro_palm_handles]
        palm_pos = self._palm_state[..., 0:3]  # [num_envs, num_arms, 3]
        self._palm_rot = self._palm_state[..., 3:7]  # [num_envs, num_arms, 4]
        for arm_idx in range(self.num_arms):
            self.palm_center_pos[:, arm_idx] = palm_pos[:, arm_idx] + quat_rotate(
                self._palm_rot[:, arm_idx], self.palm_center_offset
            )

        self.fingertip_state = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.fingertip_state[:, :, 0:3]
        self.fingertip_rot = self.fingertip_state[:, :, 3:7]

        if hasattr(self, "fingertip_pos_rel_object"):
            self.fingertip_pos_rel_object_prev[:, :, :] = self.fingertip_pos_rel_object
        else:
            self.fingertip_pos_rel_object_prev = None

        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos).to(self.device)
        for arm_idx in range(self.num_arms):
            for i in range(self.num_fingertips):
                finger_idx = arm_idx * self.num_fingertips + i
                self.fingertip_pos_offset[:, finger_idx] = self.fingertip_pos[:, finger_idx] + quat_rotate(
                    self.fingertip_rot[:, finger_idx], self.fingertip_offsets[:, i]
                )

        obj_pos_repeat = self.object_pos.unsqueeze(1).repeat(1, self.num_arms * self.num_fingertips, 1)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_pos_repeat
        self.curr_fingertip_distances = torch.norm(
            self.fingertip_pos_rel_object.view(self.num_envs, self.num_arms, self.num_fingertips, -1), dim=-1
        )

        # when episode ends or target changes we reset this to -1, this will initialize it to the actual distance on the 1st frame of the episode
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )

        palm_center_repeat = self.palm_center_pos.unsqueeze(2).repeat(
            1, 1, self.num_fingertips, 1
        )  # [num_envs, num_arms, num_fingertips, 3] == [num_envs, 2, 4, 3]
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_center_repeat.view(
            self.num_envs, self.num_arms * self.num_fingertips, 3
        )  # [num_envs, num_arms * num_fingertips, 3] == [num_envs, 8, 3]

        if self.fingertip_pos_rel_object_prev is None:
            self.fingertip_pos_rel_object_prev = self.fingertip_pos_rel_object.clone()

        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + quat_rotate(
                self.object_rot, self.object_keypoint_offsets[:, i]
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_keypoint_offsets[:, i]
            )

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos

        palm_center_repeat = self.palm_center_pos.unsqueeze(2).repeat(1, 1, self.num_keypoints, 1)
        obj_kp_pos_repeat = self.obj_keypoint_pos.unsqueeze(1).repeat(1, self.num_arms, 1, 1)
        self.keypoints_rel_palm = obj_kp_pos_repeat - palm_center_repeat
        self.keypoints_rel_palm = self.keypoints_rel_palm.view(self.num_envs, self.num_arms * self.num_keypoints, 3)

        # self.keypoints_rel_palm = self.obj_keypoint_pos - palm_center_repeat.view(
        #     self.num_envs, self.num_arms * self.num_keypoints, 3
        # )

        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)

        # furthest keypoint from the goal
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values

        # this is the closest the keypoint had been to the target in the current episode (for the furthest keypoint of all)
        # make sure we initialize this value before using it for obs or rewards
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )

        if self.obs_type == "full_state":
            full_state_size, reward_obs_ofs = self.compute_full_state(self.obs_buf)
            assert (
                full_state_size == self.full_state_size
            ), f"Expected full state size {self.full_state_size}, actual: {full_state_size}"

            return self.obs_buf, reward_obs_ofs
        else:
            raise ValueError("Unkown observations type!")

    def compute_full_state(self, buf: Tensor) -> Tuple[int, int]:
        num_dofs = self.num_hand_arm_dofs * self.num_arms
        ofs: int = 0

        # dof positions
        buf[:, ofs : ofs + num_dofs] = unscale(
            self.arm_hand_dof_pos[:, :num_dofs],
            self.arm_hand_dof_lower_limits[:num_dofs],
            self.arm_hand_dof_upper_limits[:num_dofs],
        )
        ofs += num_dofs

        # dof velocities
        buf[:, ofs : ofs + num_dofs] = self.arm_hand_dof_vel[:, :num_dofs]
        ofs += num_dofs

        # palm pos
        num_palm_coords = 3 * self.num_arms
        buf[:, ofs : ofs + num_palm_coords] = self.palm_center_pos.view(self.num_envs, num_palm_coords)
        ofs += num_palm_coords

        # palm rot, linvel, ang vel
        num_palm_rot_vel_angvel = 10 * self.num_arms
        buf[:, ofs : ofs + num_palm_rot_vel_angvel] = self._palm_state[..., 3:13].reshape(
            self.num_envs, num_palm_rot_vel_angvel
        )
        ofs += num_palm_rot_vel_angvel

        # object rot, linvel, ang vel
        buf[:, ofs : ofs + 10] = self.object_state[:, 3:13]
        ofs += 10

        # fingertip pos relative to the palm of the hand
        fingertip_rel_pos_size = 3 * self.num_arms * self.num_fingertips
        buf[:, ofs : ofs + fingertip_rel_pos_size] = self.fingertip_pos_rel_palm.reshape(
            self.num_envs, fingertip_rel_pos_size
        )
        ofs += fingertip_rel_pos_size

        # keypoint distances relative to the palm of the hand
        keypoint_rel_palm_size = 3 * self.num_arms * self.num_keypoints
        buf[:, ofs : ofs + keypoint_rel_palm_size] = self.keypoints_rel_palm.reshape(
            self.num_envs, keypoint_rel_palm_size
        )
        ofs += keypoint_rel_palm_size

        # keypoint distances relative to the goal
        keypoint_rel_pos_size = 3 * self.num_keypoints
        buf[:, ofs : ofs + keypoint_rel_pos_size] = self.keypoints_rel_goal.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        ofs += keypoint_rel_pos_size

        # object scales
        buf[:, ofs : ofs + 3] = self.object_scales
        ofs += 3

        # closest distance to the furthest of all keypoints achieved so far in this episode
        buf[:, ofs : ofs + 1] = self.closest_keypoint_max_dist.unsqueeze(-1)
        # print(f"closest_keypoint_max_dist: {self.closest_keypoint_max_dist[0]}")
        ofs += 1

        # commented out for 2-hand version to minimize the number of observations
        # closest distance between a fingertip and an object achieved since last target reset
        # this should help the critic predict the anticipated fingertip reward
        # buf[:, ofs : ofs + self.num_fingertips] = self.closest_fingertip_dist
        # print(f"closest_fingertip_dist: {self.closest_fingertip_dist[0]}")
        # ofs += self.num_fingertips

        # indicates whether we already lifted the object from the table or not, should help the critic be more accurate
        buf[:, ofs : ofs + 1] = self.lifted_object.unsqueeze(-1)
        # print(f"Lifted object: {self.lifted_object[0]}")
        ofs += 1

        # this should help the critic predict the future rewards better and anticipate the episode termination
        buf[:, ofs : ofs + 1] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
        ofs += 1
        buf[:, ofs : ofs + 1] = torch.log(self.successes + 1).unsqueeze(-1)
        ofs += 1

        # actions
        # buf[:, ofs : ofs + self.num_actions] = self.actions
        # ofs += self.num_actions

        # state_str = [f"{state.item():.3f}" for state in buf[0, : self.full_state_size]]
        # print(' '.join(state_str))

        # this is where we will add the reward observation
        reward_obs_ofs = ofs
        ofs += 1

        assert ofs == self.full_state_size
        return ofs, reward_obs_ofs

    def clamp_obs(self, obs_buf: Tensor) -> None:
        if self.clamp_abs_observations > 0:
            obs_buf.clamp_(-self.clamp_abs_observations, self.clamp_abs_observations)

    def get_random_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)

        return new_rot

    def reset_target_pose(self, env_ids: Tensor) -> None:
        self._reset_target(env_ids)

        self.reset_goal_buf[env_ids] = 0
        self.near_goal_steps[env_ids] = 0
        self.closest_keypoint_max_dist[env_ids] = -1

    def reset_object_pose(self, env_ids):
        obj_indices = self.object_indices[env_ids]

        # reset object
        table_width = 1.1
        obj_x_ofs = table_width / 2 - 0.2

        left_right_random = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
        x_pos = torch.where(
            left_right_random > 0,
            obj_x_ofs * torch.ones_like(left_right_random),
            -obj_x_ofs * torch.ones_like(left_right_random),
        )

        rand_pos_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        self.root_state_tensor[obj_indices] = self.object_init_state[env_ids].clone()

        # indices 0..2 correspond to the object position
        self.root_state_tensor[obj_indices, 0:1] = x_pos + self.reset_position_noise_x * rand_pos_floats[:, 0:1]
        self.root_state_tensor[obj_indices, 1:2] = (
            self.object_init_state[env_ids, 1:2] + self.reset_position_noise_y * rand_pos_floats[:, 1:2]
        )
        self.root_state_tensor[obj_indices, 2:3] = (
            self.object_init_state[env_ids, 2:3] + self.reset_position_noise_z * rand_pos_floats[:, 2:3]
        )

        new_object_rot = self.get_random_quat(env_ids)

        # indices 3,4,5,6 correspond to the rotation quaternion
        self.root_state_tensor[obj_indices, 3:7] = new_object_rot

        self.root_state_tensor[obj_indices, 7:13] = torch.zeros_like(self.root_state_tensor[obj_indices, 7:13])

        # since we reset the object, we also should update distances between fingers and the object
        self.closest_fingertip_dist[env_ids] = -1

    def deferred_set_actor_root_state_tensor_indexed(self, obj_indices: List[Tensor]) -> None:
        self.set_actor_root_state_object_indices.extend(obj_indices)

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices: List[Tensor] = self.set_actor_root_state_object_indices
        if not object_indices:
            # nothing to set
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )

        self.set_actor_root_state_object_indices = []

    def reset_idx(self, env_ids: Tensor) -> None:
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.reset_object_pose(env_ids)

        # flattened list of arm actors that we need to reset
        arm_indices = self.arm_indices[env_ids].to(torch.int32).flatten()

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        # reset allegro hand
        delta_max = self.arm_hand_dof_upper_limits - self.hand_arm_default_dof_pos
        delta_min = self.arm_hand_dof_lower_limits - self.hand_arm_default_dof_pos

        rand_dof_floats = torch_rand_float(
            0.0, 1.0, (len(env_ids), self.num_arms * self.num_hand_arm_dofs), device=self.device
        )

        rand_delta = delta_min + (delta_max - delta_min) * rand_dof_floats

        allegro_pos = self.hand_arm_default_dof_pos + self.pos_noise_coeff * rand_delta

        self.arm_hand_dof_pos[env_ids, ...] = allegro_pos
        self.prev_targets[env_ids, ...] = allegro_pos
        self.cur_targets[env_ids, ...] = allegro_pos

        rand_vel_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs * self.num_arms), device=self.device
        )
        self.arm_hand_dof_vel[env_ids, :] = self.reset_dof_vel_noise * rand_vel_floats

        arm_indices_gym = gymtorch.unwrap_tensor(arm_indices)
        num_arm_indices: int = len(arm_indices)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets), arm_indices_gym, num_arm_indices
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), arm_indices_gym, num_arm_indices
        )

        object_indices = [self.object_indices[env_ids]]
        object_indices.extend(self._extra_object_indices(env_ids))
        self.deferred_set_actor_root_state_tensor_indexed(object_indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0

        self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
        self.true_objective[env_ids] = 0

        self.lifted_object[env_ids] = False

        # -1 here indicates that the value is not initialized
        self.closest_keypoint_max_dist[env_ids] = -1

        self.closest_fingertip_dist[env_ids] = -1

        self.near_goal_steps[env_ids] = 0

        for key in self.rewards_episode.keys():
            # print(f"{env_ids}: {key}: {self.rewards_episode[key][env_ids]}")
            self.rewards_episode[key][env_ids] = 0

        self.extras["scalars"] = dict()
        self.extras["scalars"]["success_tolerance"] = self.success_tolerance

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        self.reset_target_pose(reset_goal_env_ids)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.set_actor_root_state_tensor_indexed()

        if self.use_relative_control:
            raise NotImplementedError("Use relative control False for now")
        else:
            # TODO: this uses simplified finger control compared to the original code of 1-hand env

            num_dofs: int = self.num_hand_arm_dofs * self.num_arms

            # target position control for the hand DOFs
            self.cur_targets[..., :num_dofs] = scale(
                actions[..., :num_dofs],
                self.arm_hand_dof_lower_limits[:num_dofs],
                self.arm_hand_dof_upper_limits[:num_dofs],
            )
            self.cur_targets[..., :num_dofs] = (
                self.act_moving_average * self.cur_targets[..., :num_dofs]
                + (1.0 - self.act_moving_average) * self.prev_targets[..., :num_dofs]
            )
            self.cur_targets[..., :num_dofs] = tensor_clamp(
                self.cur_targets[..., :num_dofs],
                self.arm_hand_dof_lower_limits[:num_dofs],
                self.arm_hand_dof_upper_limits[:num_dofs],
            )

        self.prev_targets[...] = self.cur_targets[...]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device)
                * self.object_rb_masses
                * self.force_scale
            )

            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE
            )

    def post_physics_step(self):
        self.frame_since_restart += 1

        self.progress_buf += 1
        self.randomize_buf += 1

        self._extra_curriculum()

        obs_buf, reward_obs_ofs = self.compute_observations()
        rewards, is_success = self.compute_kuka_reward()

        # add rewards to observations
        reward_obs_scale = 0.01
        obs_buf[:, reward_obs_ofs : reward_obs_ofs + 1] = rewards.unsqueeze(-1) * reward_obs_scale

        self.clamp_obs(obs_buf)

        self._eval_stats(is_success)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            axes_geom = gymutil.AxesGeometry(0.1)

            sphere_pose = gymapi.Transform()
            sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, sphere_pose, color=(1, 1, 0))
            sphere_geom_white = gymutil.WireframeSphereGeometry(0.02, 8, 8, sphere_pose, color=(1, 1, 1))

            palm_center_pos_cpu = self.palm_center_pos.cpu().numpy()
            palm_rot_cpu = self._palm_rot.cpu().numpy()

            for i in range(self.num_envs):
                palm_center_transform = gymapi.Transform()
                palm_center_transform.p = gymapi.Vec3(*palm_center_pos_cpu[i])
                palm_center_transform.r = gymapi.Quat(*palm_rot_cpu[i])
                gymutil.draw_lines(sphere_geom_white, self.gym, self.viewer, self.envs[i], palm_center_transform)

            for j in range(self.num_fingertips):
                fingertip_pos_cpu = self.fingertip_pos_offset[:, j].cpu().numpy()
                fingertip_rot_cpu = self.fingertip_rot[:, j].cpu().numpy()

                for i in range(self.num_envs):
                    fingertip_transform = gymapi.Transform()
                    fingertip_transform.p = gymapi.Vec3(*fingertip_pos_cpu[i])
                    fingertip_transform.r = gymapi.Quat(*fingertip_rot_cpu[i])

                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], fingertip_transform)

            for j in range(self.num_keypoints):
                keypoint_pos_cpu = self.obj_keypoint_pos[:, j].cpu().numpy()
                goal_keypoint_pos_cpu = self.goal_keypoint_pos[:, j].cpu().numpy()

                for i in range(self.num_envs):
                    keypoint_transform = gymapi.Transform()
                    keypoint_transform.p = gymapi.Vec3(*keypoint_pos_cpu[i])
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], keypoint_transform)

                    goal_keypoint_transform = gymapi.Transform()
                    goal_keypoint_transform.p = gymapi.Vec3(*goal_keypoint_pos_cpu[i])
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], goal_keypoint_transform)
