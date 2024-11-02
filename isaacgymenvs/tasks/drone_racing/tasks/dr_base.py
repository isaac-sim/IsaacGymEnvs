import abc
import os
import sys
from datetime import datetime
from os.path import join
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.drone_racing.drone_sim import (
    SimpleBetaflightParams,
    SimpleBetaflight,
    RotorPolyLagParams,
    RotorPolyLag,
    PropellerPolyParams,
    PropellerPoly,
    BodyDragPolyParams,
    BodyDragPoly,
    WrenchSumParams,
    WrenchSum,
)
from isaacgymenvs.tasks.drone_racing.encoders.dce import (
    VAEImageEncoder as DCEnc,
    VAEImageEncoderConfig as DCEncConfig,
)
from isaacgymenvs.tasks.drone_racing.managers import (
    CameraManager,
    RandCameraOptions,
    DroneManagerParams,
    DroneManager,
    RandDroneOptions,
)
from isaacgymenvs.tasks.drone_racing.mdp import (
    ObservationParams,
    Observation,
    RewardParams,
    Reward,
)
from isaacgymenvs.tasks.drone_racing.waypoint import (
    WaypointTrackerParams,
    WaypointTracker,
    WaypointData,
)
from isaacgymenvs.utils.torch_jit_utils import quat_rotate_inverse


class DRBase(VecTask):
    """
    Base class for vectorized drone racing task with static number of waypoints per env,
    and the assumption that drones are always reset to the center of the initial waypoint.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        rl_device: str,
        sim_device: str,
        graphics_device_id: int,
        headless: bool,
        virtual_screen_capture: bool,
        force_render: bool,
    ):
        # configurations
        self.cfg = cfg

        self.enable_debug_viz: bool = self.cfg["env"]["enableDebugVis"]
        self.enable_camera_sensors: bool = self.cfg["env"]["enableCameraSensors"]
        self.enable_virtual_walls: bool = self.cfg["env"]["enableVirtualWalls"]
        self.enable_strict_collision: bool = self.cfg["env"]["enableStrictCollision"]

        self.obs_img_mode: str = self.cfg["env"]["obsImgMode"]
        self.max_episode_length: int = self.cfg["env"]["maxEpisodeLength"]
        self.camera_width: int = self.cfg["env"]["cameraWidth"]
        self.camera_height: int = self.cfg["env"]["cameraHeight"]
        self.camera_hfov: float = self.cfg["env"]["cameraHfov"]
        self.camera_body_pos: List[float] = self.cfg["env"]["cameraBodyPos"]
        self.camera_angle_deg: float = self.cfg["env"]["cameraAngleDeg"]
        self.camera_depth_max: float = self.cfg["env"]["cameraDepthMax"]

        self.enable_logging: bool = self.cfg["env"]["logging"]["enable"]
        self.log_main_cam: bool = self.cfg["env"]["logging"]["logMainCam"]
        self.log_extra_cams: bool = self.cfg["env"]["logging"]["logExtraCams"]
        self.extra_cam_width: int = self.cfg["env"]["logging"]["extraCameraWidth"]
        self.extra_cam_height: int = self.cfg["env"]["logging"]["extraCameraHeight"]
        self.extra_cam_hfov: float = self.cfg["env"]["logging"]["extraCameraHfov"]
        self.max_log_episodes: int = self.cfg["env"]["logging"]["maxNumEpisodes"]
        self.num_steps_per_log_save: int = self.cfg["env"]["logging"]["numStepsPerSave"]
        self.log_exp_name: str = self.cfg["env"]["logging"]["experimentName"]

        self.k_vel_lateral_rew: float = self.cfg["mdp"]["extra_reward"]["k_vel_lateral"]
        self.k_vel_backward_rew: float = self.cfg["mdp"]["extra_reward"][
            "k_vel_backward"
        ]

        # create sim and envs
        self.sim: Optional[gymapi.Sim] = None
        self.num_actors_per_env: Optional[int] = None
        self.drone_actor_id_flat: Optional[torch.Tensor] = None
        self.envs: List[gymapi.Env] = []
        self.drone_actors: List[int] = []
        self.num_waypoints_to_track: Optional[int] = None
        self.env_size: Optional[float] = None
        self.enable_viewer_sync = True
        self.viewer = None
        super().__init__(
            config=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # magical warmup for proper collision checking
        self.gym.simulate(self.sim)

        # create camera sensors on drones and create image buffers
        self.camera_sensors: List[int] = []
        self.depth_image_tensors: List[torch.Tensor] = []
        self.color_image_tensors: List[torch.Tensor] = []
        self.camera_body_tf: gymapi.Transform = gymapi.Transform()
        self.camera_body_tf.p = gymapi.Vec3(*self.camera_body_pos)
        self.camera_body_tf.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), -self.camera_angle_deg * torch.pi / 180
        )

        self.extra_front_cameras: List[int] = []
        self.extra_back_cameras: List[int] = []
        self.extra_left_cameras: List[int] = []
        self.extra_right_cameras: List[int] = []
        self.extra_up_cameras: List[int] = []
        self.extra_down_cameras: List[int] = []

        self.extra_front_depth_tensors: List[torch.Tensor] = []
        self.extra_back_depth_tensors: List[torch.Tensor] = []
        self.extra_left_depth_tensors: List[torch.Tensor] = []
        self.extra_right_depth_tensors: List[torch.Tensor] = []
        self.extra_up_depth_tensors: List[torch.Tensor] = []
        self.extra_down_depth_tensors: List[torch.Tensor] = []

        if self.enable_camera_sensors:
            # camera properties
            cam_props = gymapi.CameraProperties()
            cam_props.enable_tensors = True
            cam_props.width = self.camera_width
            cam_props.height = self.camera_height
            cam_props.horizontal_fov = self.camera_hfov
            cam_props.use_collision_geometry = False  # True seems to be slower

            # extra camera properties
            extra_cam_props = gymapi.CameraProperties()
            extra_cam_props.enable_tensors = True
            extra_cam_props.width = self.extra_cam_width
            extra_cam_props.height = self.extra_cam_height
            extra_cam_props.horizontal_fov = self.extra_cam_hfov
            extra_cam_props.use_collision_geometry = False

            # create, attach cameras and allocate image buffers
            for i in tqdm(range(self.num_envs)):
                env = self.envs[i]
                drone_actor = self.drone_actors[i]

                cam = self.gym.create_camera_sensor(env, cam_props)
                self.camera_sensors.append(cam)

                self.gym.attach_camera_to_body(
                    cam, env, drone_actor, self.camera_body_tf, gymapi.FOLLOW_TRANSFORM
                )

                depth_gym_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, cam, gymapi.IMAGE_DEPTH
                )
                self.depth_image_tensors.append(gymtorch.wrap_tensor(depth_gym_tensor))

                color_gym_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, cam, gymapi.IMAGE_COLOR
                )
                self.color_image_tensors.append(gymtorch.wrap_tensor(color_gym_tensor))

                # extra cameras
                if self.log_extra_cams:
                    disable_drone_visuals = self.cfg["droneSim"]["drone_asset_options"][
                        "disable_visuals"
                    ]
                    assert (
                        self.enable_logging
                    ), "logging extra cams enabled but logging is disabled"
                    assert (
                        disable_drone_visuals
                    ), "logging extra cams requires drone visuals disabled, otherwise extra cams are blocked"

                    self._init_extra_cameras(env, drone_actor, extra_cam_props)
        else:
            self.dummy_encoded_img = torch.zeros(self.num_envs, 64, device=self.device)
            self.depth_image_batch = torch.zeros(
                self.num_envs,
                self.camera_height,
                self.camera_width,
                device=self.device,
            )

            assert (
                self.log_extra_cams is False
            ), "logging extra cams enabled but camera sensors are disabled"
            assert (
                self.log_main_cam is False
            ), "logging the main cam enabled but camera sensors are disabled"

        if self.obs_img_mode == "flat":
            assert (
                self.enable_camera_sensors
            ), "flat images cannot be in observation as camera sensors are disabled"

        # encoder
        self.dce = None
        if self.obs_img_mode == "dce":
            # if no camera enabled we output the dummy DCE vector
            assert (
                self.camera_width == 480 and self.camera_height == 270
            ), "DCE requires 480x270 input"
            assert (
                self.num_envs > 1
            ), "DCE requires the number of envs to be greater than 1, maybe a DCE bug"
            self.dce = DCEnc(DCEncConfig())
        elif self.obs_img_mode != "flat" and self.obs_img_mode != "empty":
            raise ValueError(f"expected DCE, flat, or empty, got {self.obs_img_mode}")

        # contact tensor
        self.contact_force: torch.Tensor = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        )
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # rigid body state buffers
        self.actor_root_state: torch.Tensor = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # wrench to apply buffer
        self.force_to_apply = torch.zeros(
            self.num_envs * self.num_actors_per_env, 3, device=self.device
        )
        self.torque_to_apply = torch.zeros(
            self.num_envs * self.num_actors_per_env, 3, device=self.device
        )

        # create drone sim modules
        self.simple_betaflight = SimpleBetaflight(
            self._param_from_cfg(
                SimpleBetaflightParams, self.cfg["droneSim"]["simpleBetaflight"]
            )
        )
        self.rotor_poly_lag = RotorPolyLag(
            self._param_from_cfg(
                RotorPolyLagParams, self.cfg["droneSim"]["rotorPolyLag"]
            )
        )
        self.propeller_poly = PropellerPoly(
            self._param_from_cfg(
                PropellerPolyParams, self.cfg["droneSim"]["propellerPoly"]
            )
        )
        self.body_drag_poly = BodyDragPoly(
            self._param_from_cfg(
                BodyDragPolyParams, self.cfg["droneSim"]["bodyDragPoly"]
            )
        )
        self.wrench_sum = WrenchSum(
            self._param_from_cfg(WrenchSumParams, self.cfg["droneSim"]["wrenchSum"])
        )

        # create waypoint tracker
        self.waypoint_tracker = WaypointTracker(
            WaypointTrackerParams(
                num_envs=self.num_envs,
                device=self.device,
                num_waypoints=self.num_waypoints_to_track,
            )
        )

        # create camera and drone manager
        self.camera_manager = CameraManager(
            gym=self.gym,
            cams=self.camera_sensors,
            envs=self.envs,
            drones=self.drone_actors,
            init_cam_pos=self.cfg["env"]["cameraBodyPos"],
            init_cam_angle=self.cfg["env"]["cameraAngleDeg"],
        )
        self.drone_manager = DroneManager(
            DroneManagerParams(num_envs=self.num_envs, device=self.device)
        )

        # mdp modules
        self.mdp_observation = Observation(
            self._param_from_cfg(ObservationParams, self.cfg["mdp"]["observation"])
        )
        self.mdp_reward = Reward(
            self._param_from_cfg(RewardParams, self.cfg["mdp"]["reward"])
        )

        # randomization options
        self.rand_camera_opts = self._param_from_cfg(
            RandCameraOptions, self.cfg["initRandOpt"]["randCameraOptions"]
        )
        self.rand_drone_opts = self._param_from_cfg(
            RandDroneOptions, self.cfg["initRandOpt"]["randDroneOptions"]
        )

        # pre-allocated static tensors
        self.all_env_id = torch.arange(self.num_envs, device=self.device)
        self.all_env_id_cpu = self.all_env_id.cpu()
        self.false_1d = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flu_frd = torch.tensor([[1.0, -1.0, -1.0]], device=self.device)

        # drone sim tensors
        self.actions: torch.Tensor = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )  # we need to init it for the first reset_idx
        self.actions[:, 2] = -1
        self.drone_state: Optional[torch.Tensor] = None
        self.drone_root_q: Optional[torch.Tensor] = None
        self.drone_lin_vel_w: Optional[torch.Tensor] = None
        self.drone_ang_vel_w: Optional[torch.Tensor] = None
        self.drone_lin_vel_b_frd: Optional[torch.Tensor] = None
        self.drone_ang_vel_b_frd: Optional[torch.Tensor] = None
        self.des_drone_ang_vel_b_frd: Optional[torch.Tensor] = None
        self.normalized_rotor_cmd: Optional[torch.Tensor] = None

        # waypoint info tensors
        self.waypoint_data: Optional[WaypointData] = None
        self.waypoint_passing: Optional[torch.Tensor] = None
        self.next_waypoint_id: torch.Tensor = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )  # also need it for the first reset_idx

        # common observation tensors
        self.depth_image_batch: Optional[torch.Tensor] = None
        self.flat_drone_state: Optional[torch.Tensor] = None
        self.flat_cam_pose: Optional[torch.Tensor] = None
        self.flat_waypoint_info: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None

        # reward
        self.default_reward: Optional[torch.Tensor] = None
        self.lin_vel_reward: Optional[torch.Tensor] = None

        # episode termination tensors
        self.crashed: Optional[torch.Tensor] = None
        self.finished: Optional[torch.Tensor] = None

        # flags
        self.initial_reset = False

        # logging
        self.log_data_dict: Optional[Dict[str, Any]] = None
        self.num_episodes: torch.Tensor = torch.zeros_like(self.reset_buf)
        self.phy_ang_vel_des_b_frd_buf: List[torch.Tensor] = []
        self.phy_rotor_cmd_buf: List[torch.Tensor] = []
        self.phy_position_w_buf: List[torch.Tensor] = []
        self.phy_quaternion_w_buf: List[torch.Tensor] = []
        self.phy_lin_vel_w_buf: List[torch.Tensor] = []
        self.phy_lin_vel_b_frd_buf: List[torch.Tensor] = []
        self.phy_ang_vel_b_frd_buf: List[torch.Tensor] = []
        self.log_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(sys.modules["__main__"].__file__)),
            "runs",
            "DRTask_logs",
            self.log_exp_name,
            "{date:%y-%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
        if self.enable_logging:
            os.makedirs(self.log_dir, exist_ok=True)
            self._init_log_data_dict()
            self.log_batch_id: int = 0

    def set_viewer(self):
        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames"
            )

            # read camera pose from cfg
            cfg_cam_pos: List[float] = self.cfg["env"]["viewer"]["camPos"]
            cfg_cam_target: List[float] = self.cfg["env"]["viewer"]["camTarget"]

            sim_params = self.gym.get_sim_params(self.sim)
            assert sim_params.up_axis == gymapi.UP_AXIS_Z

            self.gym.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(*cfg_cam_pos),
                gymapi.Vec3(*cfg_cam_target),
            )

    def create_sim(self):  # noqa
        # create sim
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_envs()

        # check if necessary variables are set
        assert self.num_actors_per_env is not None
        assert self.drone_actor_id_flat is not None
        assert len(self.envs) > 0
        assert len(self.drone_actors) > 0
        assert self.num_waypoints_to_track is not None
        assert self.num_waypoints_to_track > 0
        assert self.env_size > 0

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction of from algo to environment.
        Most common use case: tell the environment how far along we are in the training process.
        This is useful for implementing curriculums and things such as that.

        Keyword Args:
            rand_camera_opts: instance of ``RandCameraOptions`` for camera manager.
            rand_drone_opts: instance of ``RandDroneOptions`` for drone manager.
        """

        rand_camera_opts = kwargs.get("rand_camera_opts", None)
        rand_drone_opts = kwargs.get("rand_drone_opts", None)

        if rand_camera_opts is not None:
            self.rand_camera_opts = rand_camera_opts
        if rand_drone_opts is not None:
            self.rand_drone_opts = rand_drone_opts

    def reset(self) -> Dict[str, torch.Tensor]:
        """
        Prepare the envs for a new rollout and return the initial observation.
        Only after calling at least ``reset`` once can ``step`` be called.

        Specifically, track randomization happens here if required.
        Then common reset is called for all envs.
        """

        self._randomize_racing_tracks()
        assert self.waypoint_data is not None, "unable to reset without waypoint data"

        cam_tf_list = self.camera_manager.randomize_camera_tf(self.rand_camera_opts)

        self.waypoint_tracker.set_waypoint_data(self.waypoint_data)
        self.drone_manager.set_waypoint(self.waypoint_data)
        self.mdp_observation.set_waypoint_and_cam(self.waypoint_data, cam_tf_list)
        self.mdp_reward.set_waypoint_and_cam(self.waypoint_data, cam_tf_list)

        self.reset_idx(self.all_env_id)
        self.gym.step_graphics(self.sim)

        self._update_obs_terms()
        if self.enable_camera_sensors:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            self.depth_image_batch = (
                -torch.stack(self.depth_image_tensors) / self.camera_depth_max
            )
            self.depth_image_batch.clamp_(max=1)
            self.gym.end_access_image_tensors(self.sim)

        self._update_obs_dict()
        self._dict_tensor_to_rl_device(self.obs_dict)

        self.initial_reset = True

        return self.obs_dict

    def reset_idx(self, env_idx: torch.Tensor):
        # sample init state for reset envs
        # TODO: handle possible spawn collisions
        drone_state, action, next_wp_id = self.drone_manager.compute(
            self.rand_drone_opts, False, env_idx
        )

        # update actor root state and submit teleportation
        # if other actors are changed elsewhere, this line will submit those changes too
        self.actor_root_state[self.drone_actor_id_flat[env_idx]] = drone_state[env_idx]
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.actor_root_state)
        )
        self.gym.fetch_results(self.sim, True)

        # update action and next waypoint id
        self.actions[env_idx] = action[env_idx]
        self.next_waypoint_id[env_idx] = next_wp_id[env_idx]

        # reset modules
        self.simple_betaflight.reset(env_idx)
        self.rotor_poly_lag.reset(env_idx)
        self.waypoint_tracker.set_init_drone_state_next_wp(
            drone_state, next_wp_id, env_idx
        )
        self.mdp_observation.set_init_drone_state_action(drone_state, action, env_idx)
        self.mdp_reward.set_init_drone_state_action(drone_state, action, env_idx)

        # and don't forget to clear the progress counter
        self.progress_buf[env_idx] = 0

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # need to run reset at least once before stepping
        assert self.initial_reset, "call env reset first"

        # logging
        if self.enable_logging:
            should_exit = torch.all(self.num_episodes >= self.max_log_episodes)
            should_save = should_exit or (
                self.control_steps % self.num_steps_per_log_save == 0
                and not self.control_steps == 0
            )
            if should_save:
                torch.save(
                    self.log_data_dict,
                    os.path.join(self.log_dir, str(self.log_batch_id) + ".pt"),
                )
                self.log_batch_id += 1
                self._init_log_data_dict()  # clear saved data
                if should_exit:
                    print("stopping due to maximum episodes reached in logging mode...")
                    print("SH_IO_LOG_DIR:", self.log_dir)
                    self.cfg["env"]["logging"]["numLogFiles"] = self.log_batch_id
                    # also log waypoint data, we assume in logging mode waypoint data doesn't change
                    self.cfg["waypoint_data_p"] = self.waypoint_data.position.cpu()
                    self.cfg["waypoint_data_q"] = self.waypoint_data.quaternion.cpu()
                    self.cfg["waypoint_data_w"] = self.waypoint_data.width.cpu()
                    self.cfg["waypoint_data_h"] = self.waypoint_data.height.cpu()
                    torch.save(
                        self.cfg,
                        os.path.join(self.log_dir, "cfg.pt"),
                    )
                    sys.exit()
            self._update_log_data_pre_physics()
            self._clear_log_phy_buffers()  # for logging physics data

        # closed-loop control and physics
        self.actions = actions.clamp(-self.clip_actions, self.clip_actions)
        self.simple_betaflight.set_command(self.actions)
        self.crashed = self.false_1d
        for i in range(self.control_freq_inv):
            # update drone states
            self.drone_state = self.actor_root_state[self.drone_actor_id_flat]
            self.drone_root_q = self.drone_state[:, 3:7]  # x, y, z, w
            self.drone_lin_vel_w = self.drone_state[:, 7:10]
            self.drone_ang_vel_w = self.drone_state[:, 10:]
            self.drone_lin_vel_b_frd = self.flu_frd * quat_rotate_inverse(
                self.drone_root_q, self.drone_lin_vel_w
            )
            self.drone_ang_vel_b_frd = self.flu_frd * quat_rotate_inverse(
                self.drone_root_q, self.drone_ang_vel_w
            )

            # run drone sim modules
            self.des_drone_ang_vel_b_frd, self.normalized_rotor_cmd = (
                self.simple_betaflight.compute(self.drone_ang_vel_b_frd)
            )
            rpm, rotor_force, rotor_torque = self.rotor_poly_lag.compute(
                self.normalized_rotor_cmd
            )
            prop_force, prop_torque = self.propeller_poly.compute(rpm)
            ctrl_force, ctrl_torque = self.wrench_sum.compute(
                rotor_force + prop_force, rotor_torque + prop_torque
            )
            drag_force, drag_torque = self.body_drag_poly.compute(
                self.drone_lin_vel_b_frd, self.drone_ang_vel_b_frd
            )

            # apply force and torque
            self.force_to_apply[self.drone_actor_id_flat] = self.flu_frd * (
                ctrl_force + drag_force
            )
            self.torque_to_apply[self.drone_actor_id_flat] = self.flu_frd * (
                ctrl_torque + drag_torque
            )
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.force_to_apply),
                gymtorch.unwrap_tensor(self.torque_to_apply),
                gymapi.LOCAL_SPACE,
            )

            # log physics data
            if self.enable_logging:
                self._update_log_phy_buffers()

            # step physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.crashed = self.crashed | torch.greater(
                torch.linalg.norm(self.contact_force[self.drone_actor_id_flat], dim=1),
                0.01,  # TODO: make it configurable, Aerial Gym uses 0.05
            )

        # update if drone has crashed using virtual walls
        if self.enable_virtual_walls:
            oob = (
                (self.drone_state[:, 0].abs() > self.env_size / 2)
                | (self.drone_state[:, 1].abs() > self.env_size / 2)
                | (self.drone_state[:, 2].abs() > self.env_size)
            )
            self.crashed = self.crashed | oob

        # track waypoint
        self.drone_state = self.actor_root_state[self.drone_actor_id_flat]
        self.waypoint_passing, self.next_waypoint_id = self.waypoint_tracker.compute(
            self.drone_state
        )

        # check dones
        self.finished = torch.eq(self.next_waypoint_id, 0)
        self.timeout_buf[:] = self.progress_buf >= self.max_episode_length - 1
        if self.enable_strict_collision:
            # be careful that crashes are detected between environment steps
            # if crashed, finished and timeout should not be true
            # but if on, this will make training worse
            # TODO: self.waypoint_passing = self.waypoint_passing & ~self.crashed results in much worse training
            # TODO: what about timeout?
            self.finished = self.finished & ~self.crashed
            self.timeout_buf[:] = self.timeout_buf & ~self.crashed
        self.reset_buf[:] = self.crashed | self.finished | self.timeout_buf
        self.progress_buf += 1
        self.num_episodes += self.reset_buf

        # compute reward
        self.default_reward = self.mdp_reward.compute(
            drone_state=self.drone_state,
            action=self.actions,
            drone_collision=self.crashed,
            timeout=self.timeout_buf,
            wp_passing=self.waypoint_passing,
            next_wp_id=self.next_waypoint_id,
        )
        self._update_rew_buf()
        self._update_extra_rew_terms()

        # log data after physics
        if self.enable_logging:
            self._update_log_data_post_physics()

        # reset envs
        done_env_ids = self.reset_buf.nonzero().flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        # step rendering
        self.gym.step_graphics(self.sim)
        if self.enable_camera_sensors:
            self.gym.render_all_camera_sensors(self.sim)
        if self.force_render:
            self.render()

        # calculate useful tensors for observation
        self._update_obs_terms()
        if self.enable_camera_sensors:
            self.gym.start_access_image_tensors(self.sim)
            self.depth_image_batch = (
                -torch.stack(self.depth_image_tensors) / self.camera_depth_max
            )
            self.depth_image_batch.clamp_(max=1)
            self.gym.end_access_image_tensors(self.sim)
        self._update_obs_dict()
        self._update_extras()
        self.control_steps += 1

        self._dict_tensor_to_rl_device(self.obs_dict)
        self._dict_tensor_to_rl_device(self.extras)
        return (
            self.obs_dict,
            self.rew_buf.to(device=self.rl_device),
            self.reset_buf.to(device=self.rl_device),
            self.extras,
        )

    def render(self, mode="rgb_array"):
        """
        Overrides the base render function because we have camera running,
        so ``step_graphics`` need to happen regardless of ``enable_viewer_sync``.
        """

        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                print("stopping...")
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            # step graphics
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(
                    self.viewer,
                    join(self.record_frames_dir, f"frame_{self.control_steps}.png"),
                )

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def reset_done(self):
        raise NotImplementedError

    def pre_physics_step(self, actions: torch.Tensor):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_envs(self):
        """
        Create envs and fill in the following variables.

        - ``self.num_actors_per_env``
        - ``self.drone_actor_id_flat``
        - ``self.envs``
        - ``self.drone_actors``
        - ``self.num_waypoints_to_track``
        - ``self.env_size``
        """

        pass

    @abc.abstractmethod
    def _randomize_racing_tracks(self):
        """
        Randomize racing tracks (waypoints, obstacles...) if necessary.
        Responsible for making sure that ``self.waypoint_data`` is not ``None``.
        """

        pass

    @abc.abstractmethod
    def _update_rew_buf(self):
        """
        Update ``self.rew_buf``.
        """

        pass

    @abc.abstractmethod
    def _update_extra_rew_terms(self):
        """
        Update ``self.extras`` with reward terms.
        """

        pass

    @abc.abstractmethod
    def _update_obs_dict(self):
        """
        Update ``self.obs_dict``.
        """

        pass

    @abc.abstractmethod
    def _update_extras(self):
        """
        Update ``self.extras``.
        """

        pass

    def _init_extra_cameras(self, env, drone_actor, extra_cam_props):
        extra_front_cam = self.gym.create_camera_sensor(env, extra_cam_props)
        extra_back_cam = self.gym.create_camera_sensor(env, extra_cam_props)
        extra_left_cam = self.gym.create_camera_sensor(env, extra_cam_props)
        extra_right_cam = self.gym.create_camera_sensor(env, extra_cam_props)
        extra_up_cam = self.gym.create_camera_sensor(env, extra_cam_props)
        extra_down_cam = self.gym.create_camera_sensor(env, extra_cam_props)

        self.extra_front_cameras.append(extra_front_cam)
        self.extra_back_cameras.append(extra_back_cam)
        self.extra_left_cameras.append(extra_left_cam)
        self.extra_right_cameras.append(extra_right_cam)
        self.extra_up_cameras.append(extra_up_cam)
        self.extra_down_cameras.append(extra_down_cam)

        extra_front_cam_body_tf: gymapi.Transform = gymapi.Transform()
        extra_back_cam_body_tf: gymapi.Transform = gymapi.Transform()
        extra_left_cam_body_tf: gymapi.Transform = gymapi.Transform()
        extra_right_cam_body_tf: gymapi.Transform = gymapi.Transform()
        extra_up_cam_body_tf: gymapi.Transform = gymapi.Transform()
        extra_down_cam_body_tf: gymapi.Transform = gymapi.Transform()

        extra_back_cam_body_tf.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), torch.pi
        )
        extra_left_cam_body_tf.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), torch.pi / 2
        )
        extra_right_cam_body_tf.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), -torch.pi / 2
        )
        extra_up_cam_body_tf.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), -torch.pi / 2
        )
        extra_down_cam_body_tf.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), torch.pi / 2
        )

        self.gym.attach_camera_to_body(
            extra_front_cam,
            env,
            drone_actor,
            extra_front_cam_body_tf,
            gymapi.FOLLOW_TRANSFORM,
        )
        self.gym.attach_camera_to_body(
            extra_back_cam,
            env,
            drone_actor,
            extra_back_cam_body_tf,
            gymapi.FOLLOW_TRANSFORM,
        )
        self.gym.attach_camera_to_body(
            extra_left_cam,
            env,
            drone_actor,
            extra_left_cam_body_tf,
            gymapi.FOLLOW_TRANSFORM,
        )
        self.gym.attach_camera_to_body(
            extra_right_cam,
            env,
            drone_actor,
            extra_right_cam_body_tf,
            gymapi.FOLLOW_TRANSFORM,
        )
        self.gym.attach_camera_to_body(
            extra_up_cam,
            env,
            drone_actor,
            extra_up_cam_body_tf,
            gymapi.FOLLOW_TRANSFORM,
        )
        self.gym.attach_camera_to_body(
            extra_down_cam,
            env,
            drone_actor,
            extra_down_cam_body_tf,
            gymapi.FOLLOW_TRANSFORM,
        )

        self.extra_front_depth_tensors.append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, extra_front_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.extra_back_depth_tensors.append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, extra_back_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.extra_left_depth_tensors.append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, extra_left_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.extra_right_depth_tensors.append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, extra_right_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.extra_up_depth_tensors.append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, extra_up_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.extra_down_depth_tensors.append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, extra_down_cam, gymapi.IMAGE_DEPTH
                )
            )
        )

    def _init_log_data_dict(self):
        data_keys = [
            # pre physics
            "env_step",
            "episode_id",
            "episode_progress",
            "main_depth",
            "main_color",
            "extra_depth",
            "min_dist_to_obstacle",
            "main_cam_pose",
            "action",
            "next_waypoint_p",
            # inner physics loop data, stacked after physics
            "ang_vel_des_b_frd",
            "rotor_cmd",
            "position_w",
            "quaternion_w",
            "lin_vel_w",
            "lin_vel_b_frd",
            "ang_vel_b_frd",
            # reset mode after physics
            "is_finished",
            "is_crashed",
            "is_timeout",
        ]
        self.log_data_dict = {
            **{key: [] for key in data_keys},
        }

    def _update_log_data_pre_physics(self):
        self.log_data_dict["env_step"].append(self.control_steps)
        self.log_data_dict["episode_id"].append(self.num_episodes.clone().cpu())
        self.log_data_dict["episode_progress"].append(self.progress_buf.clone().cpu())
        # TODO: self.gym.start_access_image_tensors? it looks fine without it here... Why?
        if self.log_main_cam:
            main_depth = (
                (-torch.stack(self.depth_image_tensors)) / self.camera_depth_max
            ).nan_to_num_(posinf=0.0)
            main_depth[main_depth > 1.0] = 0.0
            self.log_data_dict["main_depth"].append(main_depth.cpu())
            self.log_data_dict["main_color"].append(
                torch.stack(self.color_image_tensors).cpu()
            )
        if self.log_extra_cams:
            extra_front_depth = -torch.stack(self.extra_front_depth_tensors)
            extra_back_depth = -torch.stack(self.extra_back_depth_tensors)
            extra_left_depth = -torch.stack(self.extra_left_depth_tensors)
            extra_right_depth = -torch.stack(self.extra_right_depth_tensors)
            extra_up_depth = -torch.stack(self.extra_up_depth_tensors)
            extra_down_depth = -torch.stack(self.extra_down_depth_tensors)
            extra_depth = torch.stack(
                [
                    extra_front_depth,
                    extra_back_depth,
                    extra_left_depth,
                    extra_right_depth,
                    extra_up_depth,
                    extra_down_depth,
                ],
                1,
            )
            min_d_to_obst = torch.min(extra_depth.view(self.num_envs, -1), 1).values
            self.log_data_dict["extra_depth"].append(
                extra_depth.nan_to_num_(posinf=0.0).cpu()
            )
            self.log_data_dict["min_dist_to_obstacle"].append(min_d_to_obst.cpu())
        # TODO: self.gym.end_access_image_tensors?
        self.log_data_dict["main_cam_pose"].append(self.flat_cam_pose.clone().cpu())
        self.log_data_dict["action"].append(self.actions.clone().cpu())
        self.log_data_dict["next_waypoint_p"].append(
            self.waypoint_data.position[
                self.all_env_id_cpu, self.next_waypoint_id.cpu()
            ]
        )

    def _update_log_data_post_physics(self):
        self.log_data_dict["ang_vel_des_b_frd"].append(
            torch.stack(self.phy_ang_vel_des_b_frd_buf).cpu()
        )
        self.log_data_dict["rotor_cmd"].append(
            torch.stack(self.phy_rotor_cmd_buf).cpu()
        )
        self.log_data_dict["position_w"].append(
            torch.stack(self.phy_position_w_buf).cpu()
        )
        self.log_data_dict["quaternion_w"].append(
            torch.stack(self.phy_quaternion_w_buf).cpu()
        )
        self.log_data_dict["lin_vel_w"].append(
            torch.stack(self.phy_lin_vel_w_buf).cpu()
        )
        self.log_data_dict["lin_vel_b_frd"].append(
            torch.stack(self.phy_lin_vel_b_frd_buf).cpu()
        )
        self.log_data_dict["ang_vel_b_frd"].append(
            torch.stack(self.phy_ang_vel_b_frd_buf).cpu()
        )

        is_crashed = self.crashed.clone().cpu()
        is_timeout = self.timeout_buf.cpu() & ~is_crashed
        is_finished = self.finished.cpu() & ~is_crashed & ~is_timeout
        self.log_data_dict["is_finished"].append(is_finished)
        self.log_data_dict["is_crashed"].append(is_crashed)
        self.log_data_dict["is_timeout"].append(is_timeout)
        if (is_finished.int() + is_crashed.int() + is_timeout.int() > 1).any():
            print(is_crashed.int())
            print(is_timeout.int())
            print(is_finished.int())
            raise ValueError("termination mode is ambiguous")

    def _clear_log_phy_buffers(self):
        self.phy_ang_vel_des_b_frd_buf.clear()
        self.phy_rotor_cmd_buf.clear()
        self.phy_position_w_buf.clear()
        self.phy_quaternion_w_buf.clear()
        self.phy_lin_vel_w_buf.clear()
        self.phy_lin_vel_b_frd_buf.clear()
        self.phy_ang_vel_b_frd_buf.clear()

    def _update_log_phy_buffers(self):
        self.phy_ang_vel_des_b_frd_buf.append(
            self.des_drone_ang_vel_b_frd.clone().cpu()
        )
        self.phy_rotor_cmd_buf.append(self.normalized_rotor_cmd.clone())
        self.phy_position_w_buf.append(self.drone_state[:, :3].clone())
        self.phy_quaternion_w_buf.append(self.drone_root_q.clone())
        self.phy_lin_vel_w_buf.append(self.drone_lin_vel_w.clone())
        self.phy_lin_vel_b_frd_buf.append(self.drone_lin_vel_b_frd.clone())
        self.phy_ang_vel_b_frd_buf.append(self.drone_ang_vel_b_frd.clone())

    def _param_from_cfg(self, param_class, cfg_dict: dict):
        p = param_class()
        for key in cfg_dict.keys():
            assert hasattr(p, key), f"{p}, {key}"
            setattr(p, key, cfg_dict[key])
        if hasattr(p, "device"):
            p.device = self.device
        if hasattr(p, "num_envs"):
            p.num_envs = self.num_envs
        return p

    def _update_obs_terms(self):
        (
            self.flat_drone_state,
            self.flat_cam_pose,
            self.flat_waypoint_info,
            self.last_action,
        ) = self.mdp_observation.compute(
            drone_state=self.actor_root_state[self.drone_actor_id_flat],
            next_wp_id=self.next_waypoint_id,
            action=self.actions,
        )

    def _dict_tensor_to_rl_device(self, dict_tensor: Dict[str, torch.Tensor]):
        for key in dict_tensor:
            dict_tensor[key] = dict_tensor[key].to(device=self.rl_device)
