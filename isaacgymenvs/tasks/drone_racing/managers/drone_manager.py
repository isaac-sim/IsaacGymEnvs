from dataclasses import dataclass
from typing import Tuple, Optional

import torch

from isaacgym import torch_utils
from isaacgymenvs.utils.torch_jit_utils import quat_rotate, quaternion_to_matrix
from ..waypoint import WaypointData


@dataclass
class DroneManagerParams:
    num_envs: int = 64
    device: str = "cuda"


@dataclass
class RandDroneOptions:
    # if min not specified and noted, it means min = -max

    # if -1, will be set to the max allowed value
    # min = 0, max value is included
    next_wp_id_max: int = 1

    dist_along_line_min: float = 0.0
    dist_along_line_max: float = 0.1
    drone_rotation_x_max: float = 1.57
    dist_to_line_max: float = 1.0  # min = 0

    # linear velocity in body frame [m/s]
    lin_vel_x_max: float = 1.0
    lin_vel_y_max: float = 1.0
    lin_vel_z_max: float = 1.0

    # angular velocity in body frame [rad/s]
    ang_vel_x_max: float = 1.0
    ang_vel_y_max: float = 1.0
    ang_vel_z_max: float = 1.0

    # cmd range [-1, 1]
    aileron_max: float = 0.2
    elevator_max: float = 0.2
    rudder_max: float = 0.2
    throttle_min: float = -1.0
    throttle_max: float = -0.5


class DroneManager:

    def __init__(self, params: DroneManagerParams):
        self.params = params
        self.all_env_id = torch.arange(params.num_envs, device=params.device)

        self.drone_state = torch.zeros(params.num_envs, 13, device=params.device)
        self.drone_state[:, 6] = 1
        self.init_cmd = torch.zeros(params.num_envs, 4, device=params.device)

        self.wp_position: Optional[torch.Tensor] = None
        self.wp_quaternion: Optional[torch.Tensor] = None
        self.wp_psi: Optional[torch.Tensor] = None
        self.wp_theta: Optional[torch.Tensor] = None
        self.wp_num: Optional[int] = None

        self.next_wp_id = torch.ones(
            params.num_envs, dtype=torch.long, device=params.device
        )

    def set_waypoint(self, wp_data: WaypointData):
        self.wp_position = wp_data.position.to(self.params.device)
        self.wp_quaternion = wp_data.quaternion.to(self.params.device)
        self.wp_psi = wp_data.psi.to(self.params.device)
        self.wp_theta = wp_data.theta.to(self.params.device)
        self.wp_num = wp_data.num_waypoints

    def compute(
        self,
        rand_drone_opts: RandDroneOptions,
        force_wp_center: bool = True,
        env_id: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute full state tensor of drones in world frame, initial commands and next waypoint.
        Only partial values selected by ``env_id`` in the returned tensors are meaningful.

        The drone is initialized within cylinder spaces between waypoints.
        For a multi-waypoint track, there are multiple cylinder spaces.

        Firstly, the next waypoint id ``n`` is randomly selected in set {1, 2, ..., ``num_waypoints`` - 1}.
        This determines within which cylinder space the drone will be spawned.

        Secondly, 3 parameters ``d``, ``phi_p``, ``r``, are sampled to determine the position of the drone.
        If ``restrict_to_wp`` is ``True``, ``d`` and ``r`` are set to zero in a post-processing step.

        There are 2 modes to initialize the orientation: same as the starting waypoint,
        and random orientation with body x aligned with the line connecting the starting and terminal waypoints.
        So thirdly, the mode integer ``m`` and rotation ``phi_x`` are sampled to determine the orientation.

        Finally, velocities in body frame and initial commands are sampled.

        To vectorize the operation, the parameters are sampled all together,
        actual pose and velocities are then calculated from the parameters.

        Args:
            env_id: ids of envs that need reset.
            force_wp_center: if True, force init at waypoint center.
            rand_drone_opts: options for random drones initial states.

        Returns:
            - Full state tensor of drone actors in world frame (num_envs, 13).
            - Initial commands in (num_envs, 4).
            - Next waypoint id in (num_envs, ).
        """

        if env_id is None:
            env_id = self.all_env_id
        num_envs_to_reset = env_id.shape[0]

        # sample the next waypoint id and orientation mode
        wp_id_max = rand_drone_opts.next_wp_id_max
        assert wp_id_max > 0
        if wp_id_max == -1:
            wp_id_max = self.wp_num - 2
        n = torch.randint(
            1,
            wp_id_max + 1,
            (num_envs_to_reset,),
            device=self.params.device,
        )
        m = torch.randint(0, 2, (num_envs_to_reset,), device=self.params.device)

        # sample other parameters
        p_min = torch.tensor(
            [
                rand_drone_opts.dist_along_line_min,  # [0] d
                -torch.pi,  # [1] phi_p
                0.0,  # [2] r
                -rand_drone_opts.drone_rotation_x_max,  # [3] phi_x
                -rand_drone_opts.lin_vel_x_max,  # [4] vx
                -rand_drone_opts.lin_vel_y_max,  # [5] vy
                -rand_drone_opts.lin_vel_z_max,  # [6] vz
                -rand_drone_opts.ang_vel_x_max,  # [7] wx
                -rand_drone_opts.ang_vel_y_max,  # [8] wy
                -rand_drone_opts.ang_vel_z_max,  # [9] wz
                -rand_drone_opts.aileron_max,  # [10] a
                -rand_drone_opts.elevator_max,  # [11] e
                rand_drone_opts.throttle_min,  # [12] t
                -rand_drone_opts.rudder_max,  # [13] r
            ],
            device=self.params.device,
        )
        p_max = torch.tensor(
            [
                rand_drone_opts.dist_along_line_max,  # [0] d
                torch.pi,  # [1] phi_p
                rand_drone_opts.dist_to_line_max,  # [2] r
                rand_drone_opts.drone_rotation_x_max,  # [3] phi_x
                rand_drone_opts.lin_vel_x_max,  # [4] vx
                rand_drone_opts.lin_vel_y_max,  # [5] vy
                rand_drone_opts.lin_vel_z_max,  # [6] vz
                rand_drone_opts.ang_vel_x_max,  # [7] wx
                rand_drone_opts.ang_vel_y_max,  # [8] wy
                rand_drone_opts.ang_vel_z_max,  # [9] wz
                rand_drone_opts.aileron_max,  # [10] a
                rand_drone_opts.elevator_max,  # [11] e
                rand_drone_opts.throttle_max,  # [12] t
                rand_drone_opts.rudder_max,  # [13] r
            ],
            device=self.params.device,
        )
        if force_wp_center:
            p_min[[0, 2]] = 0
            m.zero_()

        p = (
            torch.rand(num_envs_to_reset, 14, device=self.params.device)
            * (p_max - p_min)
            + p_min
        )

        # update drone position
        # TODO: MOST POSITIONS ARE AT RIGHT HAND SIDE (BUG?)
        n_1 = n - 1
        starting_wp_pos = self.wp_position[env_id, n_1]
        terminal_wp_pos = self.wp_position[env_id, n]
        d_vec = terminal_wp_pos - starting_wp_pos
        quat_for_drone_pos = torch_utils.quat_mul(
            self.wp_quaternion[env_id, n_1],
            torch_utils.quat_from_euler_xyz(
                p[:, 1],
                -self.wp_theta[env_id, n_1],
                self.wp_psi[env_id, n_1],
            ),
        )
        mat_for_drone_pos = quaternion_to_matrix(quat_for_drone_pos.roll(1))
        r_vec = mat_for_drone_pos[:, 1]
        self.drone_state[env_id, :3] = (
            starting_wp_pos
            + p[:, 0].unsqueeze(1) * d_vec
            + p[:, 2].unsqueeze(1) * r_vec
        )

        # update drone quaternion
        id_mode_0 = torch.nonzero(torch.eq(m, 0)).flatten()
        id_mode_1 = torch.nonzero(torch.eq(m, 1)).flatten()
        env_id_mode_0 = env_id[id_mode_0]
        env_id_mode_1 = env_id[id_mode_1]
        n_1_mode_0 = n_1[id_mode_0]
        n_1_mode_1 = n_1[id_mode_1]
        phi_x_mode_1 = p[id_mode_1, 3]
        self.drone_state[env_id_mode_0, 3:7] = self.wp_quaternion[
            env_id_mode_0, n_1_mode_0
        ]
        self.drone_state[env_id_mode_1, 3:7] = torch_utils.quat_mul(
            self.wp_quaternion[env_id_mode_1, n_1_mode_1],
            torch_utils.quat_from_euler_xyz(
                phi_x_mode_1,
                -self.wp_theta[env_id_mode_1, n_1_mode_1],
                self.wp_psi[env_id_mode_1, n_1_mode_1],
            ),
        )

        # update drone velocity (note: should be in world frame)
        self.drone_state[env_id, 7:] = p[:, 4:10]
        self.drone_state[env_id, 7:10] = quat_rotate(
            self.drone_state[env_id, 3:7], self.drone_state[env_id, 7:10]
        )

        # update init command and next waypoint id
        self.init_cmd[env_id] = p[:, -4:]
        self.next_wp_id[env_id] = n

        return self.drone_state, self.init_cmd, self.next_wp_id
