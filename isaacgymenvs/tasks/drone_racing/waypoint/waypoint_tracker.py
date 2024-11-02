from dataclasses import dataclass
from typing import Tuple

import torch

from isaacgymenvs.utils.torch_jit_utils import quaternion_to_matrix
from .waypoint_data import WaypointData


@dataclass
class WaypointTrackerParams:
    # number of parallel envs
    num_envs: int = 64

    # device to run tensor
    device: str = "cuda"

    # number of waypoints to track
    # with the default observation, this number = total waypoint number - 1
    # as the observation needs info of two future waypoints
    # and the episode finishes at waypoint[-2]
    num_waypoints: int = 3


class WaypointTracker:

    def __init__(self, params: WaypointTrackerParams):
        self.params = params
        self.all_env_id = torch.arange(self.params.num_envs)
        self.all_wp_id_un_sq = torch.arange(
            self.params.num_waypoints, device=params.device
        ).unsqueeze(0)

        self.wp_pos = torch.zeros(
            params.num_envs, params.num_waypoints, 3, device=params.device
        )
        self.wp_x_axis = torch.zeros(
            params.num_envs, params.num_waypoints, 3, device=params.device
        )
        self.wp_y_axis = torch.zeros(
            params.num_envs, params.num_waypoints, 3, device=params.device
        )
        self.wp_z_axis = torch.zeros(
            params.num_envs, params.num_waypoints, 3, device=params.device
        )
        self.wp_y_dim = torch.zeros(
            params.num_envs, params.num_waypoints, device=params.device
        )
        self.wp_z_dim = torch.zeros(
            params.num_envs, params.num_waypoints, device=params.device
        )

        self.is_wp_passed = torch.zeros(
            params.num_envs,
            params.num_waypoints,
            dtype=torch.bool,
            device=params.device,
        )
        self.last_drone_pos = torch.zeros(params.num_envs, 1, 3, device=params.device)

    def set_waypoint_data(self, wp_data: WaypointData):
        """
        Extract waypoint information from waypoint data.
        This should be called before the first call of ``compute``,
        and whenever waypoint data needs to be updated.

        Args:
            wp_data: object of ``WaypointData``.
        """

        assert wp_data.num_waypoints > self.params.num_waypoints
        assert wp_data.num_envs == self.params.num_envs

        self.wp_pos[:] = wp_data.position[:, : self.params.num_waypoints].to(
            device=self.params.device
        )

        wp_q = wp_data.quaternion[:, : self.params.num_waypoints].to(
            device=self.params.device
        )
        wp_mat = quaternion_to_matrix(wp_q.roll(1, dims=2))
        self.wp_x_axis[:] = wp_mat[:, :, :, 0]
        self.wp_y_axis[:] = wp_mat[:, :, :, 1]
        self.wp_z_axis[:] = wp_mat[:, :, :, 2]

        self.wp_y_dim[:] = wp_data.width[:, : self.params.num_waypoints].to(
            device=self.params.device
        )
        self.wp_z_dim[:] = wp_data.height[:, : self.params.num_waypoints].to(
            device=self.params.device
        )

    def set_init_drone_state_next_wp(
        self,
        drone_state: torch.Tensor,
        next_wp_id: torch.Tensor,
        env_id: torch.Tensor = None,
    ):
        """
        Sets initial drone positions and the next waypoint for all or selected envs.
        This should be called before running ``compute`` for the first time,
        and whenever drone positions have been reset.

        Args:
            drone_state: full drone state tensor in (num_envs, 13).
            next_wp_id: next waypoint id in (num_envs, ).
            env_id: 1-dim int tensor.
        """

        # by default all envs are selected
        if env_id is None:
            env_id = self.all_env_id

        # set last drone pos as the init drone pos
        self.last_drone_pos[env_id] = drone_state[env_id, :3].unsqueeze(1)

        # set the waypoint passing flag tensor using next wp id
        wp_passed = self.all_wp_id_un_sq < next_wp_id[env_id].unsqueeze(1)
        self.is_wp_passed[env_id] = wp_passed

    def compute(self, drone_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Checks waypoint passing and computes the next target waypoint id for all envs,
        based on the updated drone state and last drone state stored internally.
        This function is called during rollout.
        Make sure to call ``set_waypoint_data`` and ``set_init_drone_pos`` properly
        before running this function.

        Args:
            drone_state: full drone state tensor in (num_envs, 13).

        Returns:
            - Waypoint passing flag in (num_envs, ).
            - Next target waypoint id in (num_envs, ).
        """

        self.last_drone_pos[:], self.is_wp_passed[:], wp_passing, next_wp_id = (
            _compute_script(
                wp_pos=self.wp_pos,
                wp_x_axis=self.wp_x_axis,
                wp_y_axis=self.wp_y_axis,
                wp_z_axis=self.wp_z_axis,
                wp_y_dim=self.wp_y_dim,
                wp_z_dim=self.wp_z_dim,
                is_wp_passed=self.is_wp_passed,
                last_drone_pos=self.last_drone_pos,
                drone_state=drone_state,
            )
        )

        return wp_passing, next_wp_id


@torch.jit.script
def _compute_script(
    wp_pos: torch.Tensor,
    wp_x_axis: torch.Tensor,
    wp_y_axis: torch.Tensor,
    wp_z_axis: torch.Tensor,
    wp_y_dim: torch.Tensor,
    wp_z_dim: torch.Tensor,
    is_wp_passed: torch.Tensor,
    last_drone_pos: torch.Tensor,
    drone_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # expand drone position dim and calculate some pos diffs (num_envs, num_waypoints, 3)
    drone_pos = drone_state[:, :3].unsqueeze(1)
    drone_pos_diff = drone_pos - last_drone_pos
    last_drone_pos_to_wp = wp_pos - last_drone_pos

    # calculate intersection point param (num_envs, num_waypoints, 1)
    intersect_t_num = torch.sum(last_drone_pos_to_wp * wp_x_axis, dim=-1, keepdim=True)
    intersect_t_den = torch.sum(drone_pos_diff * wp_x_axis, dim=-1, keepdim=True)
    intersect_t = intersect_t_num / intersect_t_den

    # intersection point positions (num_envs, num_waypoints, 3)
    intersect_p = last_drone_pos + intersect_t * drone_pos_diff

    # vector from waypoint center to intersection point (num_envs, num_waypoints, 3)
    wp_to_intersect = intersect_p - wp_pos

    # project wp to intersect to y and z axes (num_envs, num_waypoints)
    intersect_proj_y = torch.sum(wp_to_intersect * wp_y_axis, dim=-1)
    intersect_proj_z = torch.sum(wp_to_intersect * wp_z_axis, dim=-1)

    # waypoint passing conditions (num_envs, num_waypoints)
    cond_dir = intersect_t_den.squeeze() > 0

    intersect_t_sq = intersect_t.squeeze()
    cont_t_nan = ~torch.isnan(intersect_t_sq)
    cond_t_lb = intersect_t_sq >= 0
    cond_t_ub = intersect_t_sq < 1

    cond_y_dim = intersect_proj_y.abs() < wp_y_dim / 2
    cond_z_dim = intersect_proj_z.abs() < wp_z_dim / 2

    cond_previous = is_wp_passed.roll(1, dims=1)
    cond_previous[:, 0] = True

    is_wp_passed_new = is_wp_passed | (
        cond_dir
        & cont_t_nan
        & cond_t_lb
        & cond_t_ub
        & cond_y_dim
        & cond_z_dim
        & cond_previous
    )

    # calculate wp passing indicator
    wp_passing = (is_wp_passed != is_wp_passed_new).any(dim=1)

    # calculate next waypoint id (num_envs, )
    next_wp_id = torch.eq(torch.cumsum(~is_wp_passed_new, dim=1), 1).max(dim=1).indices

    return drone_pos, is_wp_passed_new, wp_passing, next_wp_id
