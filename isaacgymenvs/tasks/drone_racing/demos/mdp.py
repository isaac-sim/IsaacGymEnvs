import inspect
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.drone_racing.assets import (
    create_drone_quadcopter,
    DroneQuadcopterOptions,
)
from isaacgymenvs.tasks.drone_racing.mdp import (
    RewardParams,
    Reward,
    ObservationParams,
    Observation,
)
from isaacgymenvs.tasks.drone_racing.waypoint import (
    WaypointGeneratorParams,
    RandWaypointOptions,
    WaypointGenerator,
    WaypointTrackerParams,
    WaypointTracker,
)
from isaacgymenvs.utils.torch_jit_utils import (
    quaternion_to_matrix,
    quat_from_euler_xyz,
    quat_mul,
)

print("Importing torch...")
import torch  # noqa


@dataclass
class MdpParams:
    env_size: int = 40
    num_envs: int = 4
    init_move_inc: float = 0.5
    init_rot_inc: float = torch.pi / 8
    add_ground: bool = True
    print_obs: bool = True

    quad_init_pose: gymapi.Transform = gymapi.Transform()
    quad_init_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)

    quad_asset_opts: DroneQuadcopterOptions = DroneQuadcopterOptions()
    quad_asset_opts.asset_options.fix_base_link = True

    camera_props: gymapi.CameraProperties = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = 640
    camera_props.height = 480
    camera_props.horizontal_fov = 90

    cam_tf: gymapi.Transform = gymapi.Transform()
    cam_tf.p = gymapi.Vec3(0.08, 0.0, 0.02)
    cam_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-15.0))

    wp_gen_params: WaypointGeneratorParams = WaypointGeneratorParams()
    wp_gen_params.num_envs = num_envs
    wp_gen_params.fixed_waypoint_id = 0
    wp_gen_params.fixed_waypoint_position = [0.0, 0.0, 1.0]

    wp_tracker_params: WaypointTrackerParams = WaypointTrackerParams()
    wp_tracker_params.num_envs = num_envs

    reward_params: RewardParams = RewardParams()
    reward_params.num_envs = num_envs

    rand_wp_opts: RandWaypointOptions = RandWaypointOptions()
    rand_wp_opts.init_yaw_max = 0.0
    rand_wp_opts.init_roll_max = 0.0
    rand_wp_opts.init_pitch_max = 0.0
    rand_wp_opts.theta_max = torch.pi / 12
    rand_wp_opts.psi_max = torch.pi / 2
    rand_wp_opts.alpha_max = 0
    rand_wp_opts.gamma_max = torch.pi / 6
    rand_wp_opts.r_min = 2
    rand_wp_opts.r_max = 10
    rand_wp_opts.wp_size_min = 1
    rand_wp_opts.wp_size_max = 3

    observation_params: ObservationParams = ObservationParams()
    observation_params.num_envs = num_envs


class Mdp:

    def __init__(self, params: MdpParams):
        torch.set_printoptions(linewidth=130, sci_mode=False, precision=2)

        self.params = params
        self.selected_env_id = 0
        self.move_inc = params.init_move_inc
        self.rot_inc = params.init_rot_inc
        self.quad_init_pose_p = torch.tensor(
            [
                params.quad_init_pose.p.x,
                params.quad_init_pose.p.y,
                params.quad_init_pose.p.z,
            ],
            device="cuda",
        )
        self.quad_init_pose_q = torch.tensor(
            [
                params.quad_init_pose.r.x,
                params.quad_init_pose.r.y,
                params.quad_init_pose.r.z,
                params.quad_init_pose.r.w,
            ],
            device="cuda",
        )

        self.gym, self.sim = self._create_sim_gym()
        self.viewer = self._create_viewer()

        self.envs, self.cam_tensors, self.depth_tensors = self._init_envs()
        self.gym.prepare_sim(self.sim)
        self.actor_root_state = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )

        self.wp_gen = WaypointGenerator(self.params.wp_gen_params)
        self.wp_tracker = WaypointTracker(self.params.wp_tracker_params)
        self.mdp_reward = Reward(self.params.reward_params)
        self.mdp_observation = Observation(self.params.observation_params)

        self.collision = torch.zeros(
            self.params.num_envs, dtype=torch.bool, device="cuda"
        )
        self.timeout = torch.zeros(
            self.params.num_envs, dtype=torch.bool, device="cuda"
        )

    def run(self):
        mdp_initialized = False
        init_next_wp_id = torch.ones(
            self.params.num_envs, dtype=torch.int, device="cuda"
        )
        while not self.gym.query_viewer_has_closed(self.viewer):

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            img = self.cam_tensors[self.selected_env_id].cpu().numpy()
            self.gym.end_access_image_tensors(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            move, reset, reset_all = self._check_key_update_actor_state()

            if reset_all:
                wp_data = self.wp_gen.compute(self.params.rand_wp_opts)

                self.gym.clear_lines(self.viewer)
                wp_data.visualize(self.gym, self.envs, self.viewer, 1.0)

                self.wp_tracker.set_waypoint_data(wp_data)
                self.wp_tracker.set_init_drone_state_next_wp(
                    self.actor_root_state, init_next_wp_id
                )

                self.mdp_reward.set_waypoint_and_cam(
                    wp_data, [self.params.cam_tf] * self.params.num_envs
                )
                self.mdp_reward.set_init_drone_state_action(
                    self.actor_root_state,
                    torch.zeros(self.params.num_envs, 4, device="cuda"),
                )

                self.mdp_observation.set_waypoint_and_cam(
                    wp_data, [self.params.cam_tf] * self.params.num_envs
                )
                self.mdp_observation.set_init_drone_state_action(
                    self.actor_root_state,
                    torch.zeros(self.params.num_envs, 4, device="cuda"),
                )

                mdp_initialized = True

            if reset and mdp_initialized:
                self.wp_tracker.set_init_drone_state_next_wp(
                    self.actor_root_state,
                    init_next_wp_id,
                    torch.tensor([self.selected_env_id], device="cuda"),
                )
                self.mdp_reward.set_init_drone_state_action(
                    self.actor_root_state,
                    torch.zeros(self.params.num_envs, 4, device="cuda"),
                    torch.tensor([self.selected_env_id], device="cuda"),
                )
                self.mdp_observation.set_init_drone_state_action(
                    self.actor_root_state,
                    torch.zeros(self.params.num_envs, 4, device="cuda"),
                    torch.tensor([self.selected_env_id], device="cuda"),
                )

            if move:
                # if reset_all or reset is True, move is True
                # if reset_all and reset are all False, move can still happen
                self.gym.set_actor_root_state_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.actor_root_state)
                )
                if (not reset_all) and (not reset) and mdp_initialized:
                    wp_passing, next_wp_id = self.wp_tracker.compute(
                        self.actor_root_state
                    )
                    r = self.mdp_reward.compute(
                        self.actor_root_state,
                        torch.zeros(self.params.num_envs, 4, device="cuda"),
                        self.collision,
                        self.timeout,
                        wp_passing,
                        next_wp_id,
                    )
                    o_state, o_cam, o_wp, o_act = self.mdp_observation.compute(
                        drone_state=self.actor_root_state,
                        next_wp_id=next_wp_id,
                        action=torch.zeros(self.params.num_envs, 4, device="cuda"),
                    )

                    print("------------")
                    print("- next wp id:")
                    print(next_wp_id)
                    print("- reward:")
                    print(r)
                    print("- progress:")
                    print(self.mdp_reward.reward_progress)
                    print("- perception:")
                    print(self.mdp_reward.reward_perception)
                    print("- guidance:")
                    print(self.mdp_reward.reward_guidance)
                    print("- waypoint:")
                    print(self.mdp_reward.reward_waypoint)
                    if self.params.print_obs:
                        print("- obs_state:")
                        print(o_state)
                        print("- obs_wp:")
                        print(o_wp)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, str(self.selected_env_id), (8, 32), 0, 1, (0, 255, 0), 2)
            cx = int(self.params.camera_props.width / 2)
            cy = int(self.params.camera_props.height / 2)
            cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
            cv2.imshow("fpv", img)
            cv2.waitKey(1)

            self.gym.sync_frame_time(self.sim)

    def _create_sim_gym(self) -> Tuple[gymapi.Gym, gymapi.Sim]:
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        gym = gymapi.acquire_gym()
        sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.params.add_ground:
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            gym.add_ground(sim, plane_params)
        return gym, sim

    def _create_viewer(self) -> gymapi.Viewer:
        line_number = inspect.currentframe().f_back.f_lineno
        print("Control keys: read `_create_viewer` at code line", line_number)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_0, "env_0")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "env_1")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "env_2")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "env_3")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "env_4")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_5, "env_5")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_6, "env_6")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_7, "env_7")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_8, "env_8")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_9, "env_9")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "move_front")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "move_back")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "move_left")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "move_right")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "move_down")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "move_up")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "roll_left")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "roll_right")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "yaw_left")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "yaw_right")
        self.gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_LEFT_SHIFT, "pitch_down"
        )
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "pitch_up")
        self.gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_MINUS, "move_inc_down"
        )
        self.gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_EQUAL, "move_inc_up"
        )
        self.gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_COMMA, "rot_inc_down"
        )
        self.gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_PERIOD, "rot_inc_up"
        )
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ENTER, "reset_all")

        return viewer

    def _init_envs(
        self,
    ) -> Tuple[List[gymapi.Env], List[torch.Tensor], List[torch.Tensor]]:
        # asset
        quad_asset = create_drone_quadcopter(
            self.gym, self.sim, self.params.quad_asset_opts
        )

        # envs
        envs = []
        cam_tensors = []
        depth_tensors = []
        # create envs
        for i in range(self.params.num_envs):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-self.params.env_size / 2, -self.params.env_size / 2, 0),
                gymapi.Vec3(
                    self.params.env_size / 2,
                    self.params.env_size / 2,
                    self.params.env_size,
                ),
                int(self.params.num_envs**0.5),
            )
            envs.append(env)

            quad_actor = self.gym.create_actor(
                env, quad_asset, self.params.quad_init_pose, "Quadcopter", i, 1
            )

            cam = self.gym.create_camera_sensor(env, self.params.camera_props)
            self.gym.attach_camera_to_body(
                cam, env, quad_actor, self.params.cam_tf, gymapi.FOLLOW_TRANSFORM
            )
            depth_gym_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env, cam, gymapi.IMAGE_DEPTH
            )
            depth_tensors.append(gymtorch.wrap_tensor(depth_gym_tensor))

            if i < 10:
                color_gym_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, cam, gymapi.IMAGE_COLOR
                )
                cam_tensors.append(gymtorch.wrap_tensor(color_gym_tensor))

        return envs, cam_tensors, depth_tensors

    def _check_key_update_actor_state(self) -> Tuple[bool, bool, bool]:
        move = False
        reset = False
        reset_all = False

        for evt in self.gym.query_viewer_action_events(self.viewer):
            # guard against key release
            if evt.value <= 0.0:
                break

            # select env
            for env_id in range(10):
                if evt.action == "env_" + str(env_id) and env_id < self.params.num_envs:
                    self.selected_env_id = env_id
                    print("selected env", env_id)
                    break

            actor_rot_mat = quaternion_to_matrix(
                self.actor_root_state[:, 3:7].roll(1, dims=1)
            )

            # change increment
            if evt.action == "move_inc_down":
                self.move_inc /= 2
                print("move increment", self.move_inc)

            elif evt.action == "move_inc_up":
                self.move_inc *= 2
                print("move increment", self.move_inc)

            elif evt.action == "rot_inc_down":
                self.rot_inc /= 2
                print("rotation increment", self.rot_inc)

            elif evt.action == "rot_inc_up":
                self.rot_inc *= 2
                print("rotation increment", self.rot_inc)

            # move
            elif evt.action == "move_front":
                displacement = actor_rot_mat[self.selected_env_id, :, 0] * self.move_inc
                self.actor_root_state[self.selected_env_id, 0:3] += displacement
                move = True

            elif evt.action == "move_back":
                displacement = actor_rot_mat[self.selected_env_id, :, 0] * self.move_inc
                self.actor_root_state[self.selected_env_id, 0:3] -= displacement
                move = True

            elif evt.action == "move_left":
                displacement = actor_rot_mat[self.selected_env_id, :, 1] * self.move_inc
                self.actor_root_state[self.selected_env_id, 0:3] += displacement
                move = True

            elif evt.action == "move_right":
                displacement = actor_rot_mat[self.selected_env_id, :, 1] * self.move_inc
                self.actor_root_state[self.selected_env_id, 0:3] -= displacement
                move = True

            elif evt.action == "move_up":
                displacement = actor_rot_mat[self.selected_env_id, :, 2] * self.move_inc
                self.actor_root_state[self.selected_env_id, 0:3] += displacement
                move = True

            elif evt.action == "move_down":
                displacement = actor_rot_mat[self.selected_env_id, :, 2] * self.move_inc
                self.actor_root_state[self.selected_env_id, 0:3] -= displacement
                move = True

            # rotation
            elif evt.action == "roll_left":
                rotation_q = quat_from_euler_xyz(
                    torch.tensor(-self.rot_inc),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                ).to(device="cuda")
                self.actor_root_state[self.selected_env_id, 3:7] = quat_mul(
                    self.actor_root_state[self.selected_env_id, 3:7], rotation_q
                )
                move = True

            elif evt.action == "roll_right":
                rotation_q = quat_from_euler_xyz(
                    torch.tensor(self.rot_inc), torch.tensor(0.0), torch.tensor(0.0)
                ).to(device="cuda")
                self.actor_root_state[self.selected_env_id, 3:7] = quat_mul(
                    self.actor_root_state[self.selected_env_id, 3:7], rotation_q
                )
                move = True

            elif evt.action == "pitch_up":
                rotation_q = quat_from_euler_xyz(
                    torch.tensor(0.0),
                    torch.tensor(-self.rot_inc),
                    torch.tensor(0.0),
                ).to(device="cuda")
                self.actor_root_state[self.selected_env_id, 3:7] = quat_mul(
                    self.actor_root_state[self.selected_env_id, 3:7], rotation_q
                )
                move = True

            elif evt.action == "pitch_down":
                rotation_q = quat_from_euler_xyz(
                    torch.tensor(0.0), torch.tensor(self.rot_inc), torch.tensor(0.0)
                ).to(device="cuda")
                self.actor_root_state[self.selected_env_id, 3:7] = quat_mul(
                    self.actor_root_state[self.selected_env_id, 3:7], rotation_q
                )
                move = True

            elif evt.action == "yaw_left":
                rotation_q = quat_from_euler_xyz(
                    torch.tensor(0.0), torch.tensor(0.0), torch.tensor(self.rot_inc)
                ).to(device="cuda")
                self.actor_root_state[self.selected_env_id, 3:7] = quat_mul(
                    self.actor_root_state[self.selected_env_id, 3:7], rotation_q
                )
                move = True

            elif evt.action == "yaw_right":
                rotation_q = quat_from_euler_xyz(
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    torch.tensor(-self.rot_inc),
                ).to(device="cuda")
                self.actor_root_state[self.selected_env_id, 3:7] = quat_mul(
                    self.actor_root_state[self.selected_env_id, 3:7], rotation_q
                )
                move = True

            # reset, no new waypoint
            elif evt.action == "reset":
                self.actor_root_state[self.selected_env_id, :3] = self.quad_init_pose_p
                self.actor_root_state[self.selected_env_id, 3:7] = self.quad_init_pose_q
                move = True
                reset = True

            # new waypoints and reset all
            elif evt.action == "reset_all":
                self.actor_root_state[:, :3] = self.quad_init_pose_p
                self.actor_root_state[:, 3:7] = self.quad_init_pose_q
                move = True
                reset_all = True

        return move, reset, reset_all


if __name__ == "__main__":
    mdp = Mdp((MdpParams()))
    mdp.run()
