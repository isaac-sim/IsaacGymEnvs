import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import pygame
import zmq

from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.drone_racing.assets import (
    create_drone_quadcopter,
    TrackMultiStoryOptions,
    TrackRmuaOptions,
    TrackSplitsOptions,
    TrackWallsOptions,
    create_track_multistory,
    create_track_rmua,
    create_track_splits,
    create_track_walls,
)
from isaacgymenvs.tasks.drone_racing.drone_sim import (
    SimpleBetaflight,
    BodyDragPoly,
    PropellerPoly,
    RotorPolyLag,
    WrenchSum,
    Kingfisher250,
)
from isaacgymenvs.tasks.drone_racing.waypoint import WaypointData
from isaacgymenvs.utils.torch_jit_utils import quat_rotate_inverse

print("Importing torch...")
import torch  # noqa


@dataclass
class QuadFpvParams:
    """
    No arg parser for simplicity, just modify me directly.
    """

    # toggles
    show_viewer: bool = True
    enable_joysticks: bool = False
    enable_zmq: bool = True
    show_fpv: bool = True
    fpv_depth: bool = False
    # when fpv_depth is false, this decides if image is displayed in color or gray
    fpv_gray: bool = False

    # depth clipping
    depth_max: float = 50.0

    # sim settings
    compute_device_id: int = 0
    graphics_device_id: int = 0
    physics_engine: gymapi.SimType = gymapi.SIM_PHYSX
    sim_params: gymapi.SimParams = gymapi.SimParams()
    sim_params.dt = 1 / 250
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    fps: int = 50

    # env settings
    num_envs: int = 100
    env_spacing: float = 0.5
    preset: Kingfisher250 = Kingfisher250(num_envs, "cuda")

    # pygame joysticks
    joystick_channels: List[int] = field(default_factory=lambda: [1, 2, 0, 3])
    joystick_directions: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    joystick_deadzones: List[float] = field(
        default_factory=lambda: [0.01, 0.01, 0.0, 0.01]
    )

    # zmq for plot juggler
    zmq_port: int = 9872


class QuadFpv:

    def __init__(self, params: QuadFpvParams):
        """
        Init Isaac Gym FPV. This script can also be used for flight tuning with PlotJuggler.

        Args:
            params: all necessary params.
        """

        # store params
        self.params = params

        # create sim and ground plane
        self.gym = gymapi.acquire_gym()

        self.sim = self.gym.create_sim(
            params.compute_device_id,
            params.graphics_device_id,
            params.physics_engine,
            params.sim_params,
        )

        if params.show_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
            print("Key R: reset")
        else:
            self.viewer = None

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # sim scheduling
        self.num_physics_per_render = int(
            1 / self.params.fps / self.params.sim_params.dt
        )
        pygame.init()
        self.pygame_clk = pygame.time.Clock()
        self.frame_time = 1000 / self.params.fps

        # initialize envs, actors, sensors, buffers
        self.quad_asset = create_drone_quadcopter(
            self.gym, self.sim, self.params.preset.quad_asset_options
        )
        self._init_envs()
        self.gym.prepare_sim(self.sim)

        # init uav simulation
        self.simple_betaflight = SimpleBetaflight(params.preset.simple_bf_params)
        self.rotor_poly_lag = RotorPolyLag(params.preset.rotor_params)
        self.propeller_poly = PropellerPoly(params.preset.propeller_params)
        self.body_drag_poly = BodyDragPoly(params.preset.body_drag_params)
        self.wrench_sum = WrenchSum(params.preset.wrench_sum_params)
        self.actor_states = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.init_actor_states = self.actor_states.clone()
        self.flu_frd = torch.tensor([[1.0, -1.0, -1.0]], device="cuda")
        self.command = torch.zeros(self.params.num_envs, 4, device="cuda")
        self.command[:, 2] = -1

        # init pygame for joystick and fps control
        if self.params.enable_joysticks:
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

        # zmq for plot juggler
        if self.params.enable_zmq:
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind("tcp://*:" + str(self.params.zmq_port))
            self.zmq_data = torch.zeros(
                self.num_physics_per_render, 2, 3, device="cuda"
            )

    def run(self):
        """
        This function runs the sim loop.
        """

        total_force = torch.zeros(
            self.num_scene_actors + self.params.num_envs, 3, device="cuda"
        )
        total_torque = torch.zeros(
            self.num_scene_actors + self.params.num_envs, 3, device="cuda"
        )

        while self.viewer is None or not self.gym.query_viewer_has_closed(self.viewer):
            # check reset
            if self.viewer is not None:
                for evt in self.gym.query_viewer_action_events(self.viewer):
                    if evt.action == "reset" and evt.value > 0:
                        self.simple_betaflight.reset()
                        self.rotor_poly_lag.reset()
                        self.gym.set_actor_root_state_tensor(
                            self.sim, gymtorch.unwrap_tensor(self.init_actor_states)
                        )

            # commands are updated at rendering rate
            if self.params.enable_joysticks:
                self._update_command_joysticks()

            self.simple_betaflight.set_command(self.command)

            # physics steps
            t_physics_start = time.time()

            for i in range(self.num_physics_per_render):
                # get body frame linear and angular velocities in FRD
                lin_vel, ang_vel = self._get_body_vel_frd()

                # get ctrl wrench
                des_ang_vel, normalized_cmd = self.simple_betaflight.compute(ang_vel)
                if self.params.enable_zmq:
                    self.zmq_data[i, 0, :] = des_ang_vel[0]
                    self.zmq_data[i, 1, :] = ang_vel[0]

                rpm, rotor_force, rotor_torque = self.rotor_poly_lag.compute(
                    normalized_cmd
                )
                prop_force, prop_torque = self.propeller_poly.compute(rpm)
                ctrl_force, ctrl_torque = self.wrench_sum.compute(
                    rotor_force + prop_force, rotor_torque + prop_torque
                )

                # get drag wrench
                drag_force, drag_torque = self.body_drag_poly.compute(lin_vel, ang_vel)

                # apply total wrench
                total_force[self.quad_actor_id] = (
                    ctrl_force + drag_force
                ) * self.flu_frd
                total_torque[self.quad_actor_id] = (
                    ctrl_torque + drag_torque
                ) * self.flu_frd
                self.gym.apply_rigid_body_force_tensors(
                    self.sim,
                    gymtorch.unwrap_tensor(total_force),
                    gymtorch.unwrap_tensor(total_torque),
                    gymapi.LOCAL_SPACE,
                )

                # step physics
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.refresh_actor_root_state_tensor(self.sim)

            t_physics_end = time.time()
            t_physics_dur = t_physics_end - t_physics_start

            # step graphics and render cam sensors
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            # access image tensors
            self.gym.start_access_image_tensors(self.sim)
            img = self.fpv_cam_tensor.cpu().numpy()
            self.gym.end_access_image_tensors(self.sim)

            # send out debug data
            if self.params.enable_zmq:
                self._zmq_publish()

            # update viewer frame
            if self.viewer is not None:
                self.gym.draw_viewer(self.viewer, self.sim, True)

            # update fpv
            if self.params.show_fpv:
                self._update_fpv(img, t_physics_dur)

            # limit fps
            self.frame_time = self.pygame_clk.tick(self.params.fps)

        if self.params.enable_zmq:
            self.zmq_socket.close()
            self.zmq_context.term()

    def _init_envs(self):
        lb = gymapi.Vec3(-self.params.env_spacing, -self.params.env_spacing, 0)
        ub = gymapi.Vec3(
            self.params.env_spacing, self.params.env_spacing, self.params.env_spacing
        )

        self.envs = []
        self.quad_actors = []
        self.num_scene_actors = 0
        wp_lists = []
        asset_tfs = []

        for i in range(self.params.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lb, ub, int(self.params.num_envs**0.5))
            self.envs.append(env)

            # create scene actors for fpv env
            if i == 0:
                ms_tf = gymapi.Transform()
                ms_tf.p = gymapi.Vec3(40, 0, 0)
                asset_tfs.append(ms_tf)
                multistory_asset, multistory_wp = create_track_multistory(
                    self.gym, self.sim, TrackMultiStoryOptions()
                )
                self.gym.create_actor(env, multistory_asset, ms_tf, "multistory", i, 1)
                self.num_scene_actors += 1
                wp_lists.append(multistory_wp)

                rmua_tf = gymapi.Transform()
                rmua_tf.p = gymapi.Vec3(-40, 0, 0)
                asset_tfs.append(rmua_tf)
                rmua_asset, rmua_wp = create_track_rmua(
                    self.gym, self.sim, TrackRmuaOptions()
                )
                self.gym.create_actor(env, rmua_asset, rmua_tf, "rmua", i, 1)
                self.num_scene_actors += 1
                wp_lists.append(rmua_wp)

                splits_tf = gymapi.Transform()
                splits_tf.p = gymapi.Vec3(0, 40, 0)
                asset_tfs.append(splits_tf)
                splits_asset, splits_wp = create_track_splits(
                    self.gym, self.sim, TrackSplitsOptions()
                )
                self.gym.create_actor(env, splits_asset, splits_tf, "splits", i, 1)
                self.num_scene_actors += 1
                wp_lists.append(splits_wp)

                walls_tf = gymapi.Transform()
                walls_tf.p = gymapi.Vec3(0, -40, 0)
                asset_tfs.append(walls_tf)
                walls_asset, walls_wp = create_track_walls(
                    self.gym, self.sim, TrackWallsOptions()
                )
                self.gym.create_actor(env, walls_asset, walls_tf, "walls", i, 1)
                self.num_scene_actors += 1
                wp_lists.append(walls_wp)

                if self.viewer is not None:
                    for j in range(len(wp_lists)):
                        wp_data = WaypointData.from_waypoint_list(1, wp_lists[j])
                        wp_data.position += torch.tensor(
                            [asset_tfs[j].p.x, asset_tfs[j].p.y, asset_tfs[j].p.z]
                        )
                        wp_data.visualize(self.gym, [self.envs[i]], self.viewer, 1)

            # create actor
            quad_init_pose = gymapi.Transform()
            quad_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
            quad_actor = self.gym.create_actor(
                env, self.quad_asset, quad_init_pose, "Quadcopter", i, 0
            )
            self.quad_actors.append(quad_actor)

            # create camera sensor for fpv env
            if i == 0:
                # different settings if viewing depth
                image_type = gymapi.IMAGE_COLOR
                if self.params.fpv_depth:
                    self.params.preset.camera_props.use_collision_geometry = True
                    image_type = gymapi.IMAGE_DEPTH

                # create camera sensor and attach to body
                self.quad_fpv_cam = self.gym.create_camera_sensor(
                    env, self.params.preset.camera_props
                )
                self.gym.attach_camera_to_body(
                    self.quad_fpv_cam,
                    env,
                    quad_actor,
                    self.params.preset.camera_pose,
                    gymapi.FOLLOW_TRANSFORM,
                )

                # init buffer
                cam_gym_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, self.quad_fpv_cam, image_type
                )
                self.fpv_cam_tensor = gymtorch.wrap_tensor(cam_gym_tensor)

        self.quad_actor_id = torch.arange(
            self.num_scene_actors,
            self.num_scene_actors + self.params.num_envs,
            device="cuda",
        )

    def _update_command_joysticks(self):
        pygame.event.get()
        for i in range(4):
            joystick_input = self.joystick.get_axis(self.params.joystick_channels[i])
            self.command[:, i] = (
                joystick_input
                * self.params.joystick_directions[i]
                * (abs(joystick_input) > self.params.joystick_deadzones[i])
            )

    def _get_body_vel_frd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        attitude = self.actor_states[self.quad_actor_id, 3:7]
        lin_vel_w = self.actor_states[self.quad_actor_id, 7:10]
        ang_vel_w = self.actor_states[self.quad_actor_id, 10:13]
        lin_vel_b = quat_rotate_inverse(attitude, lin_vel_w) * self.flu_frd
        ang_vel_b = quat_rotate_inverse(attitude, ang_vel_w) * self.flu_frd
        return lin_vel_b, ang_vel_b

    def _update_fpv(self, img: np.ndarray, t_physics: float):
        if self.params.fpv_depth:
            # note that debug lines will also appear
            # with depth clipped, it is harder to maintain attitude awareness
            # img[img < -self.params.depth_max] = -self.params.depth_max
            # img = -img / self.params.depth_max
            img = img / self.params.depth_max + 1
            np.clip(img, 0, 1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            calculated_fps = int(1000 / self.frame_time)
            cv2.putText(
                img,
                str(calculated_fps) + ", " + str(int(t_physics * 1000)),
                (8, 32),
                0,
                1,
                (0, 255, 0),
                2,
            )
            cx = int(self.params.preset.camera_props.width / 2)
            cy = int(self.params.preset.camera_props.height / 2)
            cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
            if self.params.fpv_gray:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imshow("fpv", img)
        cv2.waitKey(1)

    def _zmq_publish(self):
        data_np = self.zmq_data.cpu().numpy()
        for i in range(data_np.shape[0]):
            msg = {
                "t": time.time() - self.params.sim_params.dt * (data_np.shape[0] - i),
                "des_ang_vel": {
                    "x": float(data_np[i, 0, 0]),
                    "y": float(data_np[i, 0, 1]),
                    "z": float(data_np[i, 0, 2]),
                },
                "ang_vel": {
                    "x": float(data_np[i, 1, 0]),
                    "y": float(data_np[i, 1, 1]),
                    "z": float(data_np[i, 1, 2]),
                },
            }
            self.zmq_socket.send_string(json.dumps(msg))


if __name__ == "__main__":
    fpv_params = QuadFpvParams()
    fpv = QuadFpv(fpv_params)
    fpv.run()
