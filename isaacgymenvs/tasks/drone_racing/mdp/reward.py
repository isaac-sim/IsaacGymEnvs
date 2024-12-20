from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch

from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import (
    quaternion_to_matrix,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)
from ..waypoint import WaypointData


@dataclass
class RewardParams:
    # number of parallel envs
    num_envs: int = 64

    # device to run tensor
    device: str = "cuda"

    # progress reward coefficient (swift: 1.0)
    k_progress: float = 1.0

    # perception reward coefficient (swift: 0.02)
    k_perception: float = 0.02

    # camera deviation coefficient (swift: -10)
    k_cam_dev: float = -10.0

    # cmd reward (ang vel part) coefficient (swift: -2e-4)
    k_cmd_ang_vel: float = -2e-4

    # cmd reward (diff) coefficient (swift: -1e-4)
    k_cmd_diff: float = -1e-4

    # collision reward coefficient (swift: -5)
    k_collision: float = -5.0

    # guidance reward coefficient
    k_guidance: float = 1.0

    # rejection coefficient
    k_rejection: float = 2.0

    # waypoint passing reward coefficient
    k_waypoint: float = 2.5

    # timeout reward coefficient
    k_timeout: float = -5.0

    # guidance x threshold
    guidance_x_thresh: float = 3.0

    # guidance tolerance
    guidance_tol: float = 1 / 8

    # enable progress reward normalization by distance
    # turning it on smoothes the progress reward when parallel envs are different
    enable_normalization: bool = False


class Reward:

    def __init__(self, params: RewardParams):
        self.params = params
        self.all_env_id = torch.arange(params.num_envs, device=params.device)

        self.cam_tf_p = torch.zeros(params.num_envs, 3, device=params.device)
        self.cam_tf_q = torch.zeros(params.num_envs, 4, device=params.device)

        self.last_dist_to_wp = torch.zeros(params.num_envs, device=params.device)
        self.last_action = torch.zeros(params.num_envs, 4, device=params.device)

        self.reward_progress = torch.zeros(params.num_envs, device=params.device)
        self.reward_perception = torch.zeros(params.num_envs, device=params.device)
        self.reward_cmd = torch.zeros(params.num_envs, device=params.device)
        self.reward_collision = torch.zeros(params.num_envs, device=params.device)
        self.reward_guidance = torch.zeros(params.num_envs, device=params.device)
        self.reward_waypoint = torch.zeros(params.num_envs, device=params.device)
        self.reward_timeout = torch.zeros(params.num_envs, device=params.device)

        self.wp_position: Optional[torch.Tensor] = None
        self.wp_quaternion: Optional[torch.Tensor] = None
        self.wp_width: Optional[torch.Tensor] = None
        self.wp_height: Optional[torch.Tensor] = None
        self.wp_r_sum: Optional[torch.Tensor] = None

    def set_waypoint_and_cam(
        self, wp_data: WaypointData, cam_tf: List[gymapi.Transform]
    ):
        assert self.params.num_envs == wp_data.num_envs

        self.wp_position = wp_data.position.to(device=self.params.device)
        self.wp_quaternion = wp_data.quaternion.to(device=self.params.device)
        self.wp_width = wp_data.width.to(device=self.params.device)
        self.wp_height = wp_data.height.to(device=self.params.device)
        self.wp_r_sum = torch.sum(
            wp_data.r[:, :-1].to(device=self.params.device), dim=-1
        )

        for i in range(len(cam_tf)):
            self.cam_tf_p[i] = torch.tensor(
                [cam_tf[i].p.x, cam_tf[i].p.y, cam_tf[i].p.z], device=self.params.device
            )
            self.cam_tf_q[i] = torch.tensor(
                [cam_tf[i].r.x, cam_tf[i].r.y, cam_tf[i].r.z, cam_tf[i].r.w],
                device=self.params.device,
            )

    def set_init_drone_state_action(
        self,
        drone_state: torch.Tensor,
        action: torch.Tensor,
        env_id: torch.Tensor = None,
    ):
        if env_id is None:
            env_id = self.all_env_id

        drone_pos = drone_state[env_id, :3]
        wp_pos = self.wp_position[env_id, 1]
        self.last_dist_to_wp[env_id] = torch.linalg.norm(wp_pos - drone_pos, dim=1)
        self.last_action[env_id] = action[env_id]

    def compute(
        self,
        drone_state: torch.Tensor,
        action: torch.Tensor,
        drone_collision: torch.Tensor,
        timeout: torch.Tensor,
        wp_passing: torch.Tensor,
        next_wp_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the total reward and updates reward terms, which can be accessed later by user.

        Args:
            drone_state: full drone state tensor in (num_envs, 13) [p, q, v, w], here w is in world frame.
            action: action tensor in (num_envs, 4).
            drone_collision: 1-dim bool tensor indicating presence of collision in (num_envs, ).
            timeout: 1-dim bool tensor indicating timeout in (num_envs, ).
            wp_passing: 1-dim bool tensor indicating waypoint passing in (num_envs, ).
            next_wp_id: 1-dim int tensor indicating next wp id in (num_envs, ).

        Returns:
            - Reward tensor for all envs in (num_envs, ).
        """

        (
            self.reward_progress[:],
            self.reward_perception[:],
            self.reward_cmd[:],
            self.reward_collision[:],
            self.reward_guidance[:],
            self.reward_waypoint[:],
            self.reward_timeout[:],
            self.last_dist_to_wp[:],
            self.last_action[:],
        ) = _compute_script(
            k_progress=self.params.k_progress,
            k_perception=self.params.k_perception,
            k_cam_dev=self.params.k_cam_dev,
            k_cmd_ang_vel=self.params.k_cmd_ang_vel,
            k_cmd_diff=self.params.k_cmd_diff,
            k_collision=self.params.k_collision,
            k_guidance=self.params.k_guidance,
            k_rejection=self.params.k_rejection,
            k_waypoint=self.params.k_waypoint,
            k_timeout=self.params.k_timeout,
            guidance_x_thresh=self.params.guidance_x_thresh,
            guidance_tol=self.params.guidance_tol,
            enable_normalization=self.params.enable_normalization,
            wp_position=self.wp_position,
            wp_quaternion=self.wp_quaternion,
            wp_width=self.wp_width,
            wp_height=self.wp_height,
            wp_r_sum=self.wp_r_sum,
            all_env_id=self.all_env_id,
            cam_tf_p=self.cam_tf_p,
            cam_tf_q=self.cam_tf_q,
            last_dist_to_wp=self.last_dist_to_wp,
            last_action=self.last_action,
            drone_state=drone_state,
            action=action,
            drone_collision=drone_collision,
            timeout=timeout,
            wp_passing=wp_passing,
            next_wp_id=next_wp_id,
        )

        reward = (
            self.reward_progress
            + self.reward_perception
            + self.reward_cmd
            + self.reward_collision
            + self.reward_guidance
            + self.reward_waypoint
            + self.reward_timeout
        )

        return reward


@torch.jit.script
def _compute_script(
    k_progress: float,
    k_perception: float,
    k_cam_dev: float,
    k_cmd_ang_vel: float,
    k_cmd_diff: float,
    k_collision: float,
    k_guidance: float,
    k_rejection: float,
    k_waypoint: float,
    k_timeout: float,
    guidance_x_thresh: float,
    guidance_tol: float,
    enable_normalization: bool,
    wp_position: torch.Tensor,
    wp_quaternion: torch.Tensor,
    wp_width: torch.Tensor,
    wp_height: torch.Tensor,
    wp_r_sum: torch.Tensor,
    all_env_id: torch.Tensor,
    cam_tf_p: torch.Tensor,
    cam_tf_q: torch.Tensor,
    last_dist_to_wp: torch.Tensor,
    last_action: torch.Tensor,
    drone_state: torch.Tensor,
    action: torch.Tensor,
    drone_collision: torch.Tensor,
    timeout: torch.Tensor,
    wp_passing: torch.Tensor,
    next_wp_id: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # progress reward
    drone_pos = drone_state[:, :3]  # (num_envs, 3)
    wp_pos = wp_position[all_env_id, next_wp_id]  # (num_envs, 3)
    dist_to_wp = torch.linalg.norm(wp_pos - drone_pos, dim=1)  # (num_envs, )

    #              wp1                              wp2
    # O--------------|------>O                        |
    # |<---last_d--->|       |<--------this_d-------->|
    #                        ^ here wp_passing = True, and next_wp_id = 2
    # set progress to 0 at wp passing step
    # to avoid undesired negative progress
    reward_progress = k_progress * (last_dist_to_wp - dist_to_wp) * (~wp_passing)

    # perception reward
    drone_q = drone_state[:, 3:7]
    cam_q = quat_mul(drone_q, cam_tf_q)
    cam_mat = quaternion_to_matrix(cam_q.roll(1, dims=1))
    cam_x_axis = cam_mat[:, :, 0]

    cam_pos = drone_pos + quat_rotate(cam_tf_q, cam_tf_p)
    vec_cam_to_wp = wp_pos - cam_pos
    dist_cam_to_wp = torch.linalg.norm(vec_cam_to_wp, dim=-1)

    cam_dev = torch.acos(torch.sum(cam_x_axis * vec_cam_to_wp, dim=-1) / dist_cam_to_wp)
    cam_dev.nan_to_num_(nan=0.0)

    reward_perception = k_perception * torch.exp(k_cam_dev * cam_dev**4)

    # angular velocity reward
    reward_cmd = k_cmd_ang_vel * torch.linalg.norm(
        action[:, [0, 1, 3]], dim=1
    ) + k_cmd_diff * torch.linalg.norm(action - last_action, dim=1)

    # collision reward
    reward_collision = k_collision * drone_collision

    # guidance reward
    # TODO: use positive guidance?
    wp_to_drone = drone_pos - wp_pos
    wp_q = wp_quaternion[all_env_id, next_wp_id]
    drone_pos_wp_frame = quat_rotate_inverse(wp_q, wp_to_drone)
    x, y, z = (
        drone_pos_wp_frame[:, 0],
        drone_pos_wp_frame[:, 1],
        drone_pos_wp_frame[:, 2],
    )
    w = wp_width[all_env_id, next_wp_id]
    h = wp_height[all_env_id, next_wp_id]

    layer_x = -torch.sign(x) / guidance_x_thresh * x + 1
    layer_x.clamp_(min=0.0)
    guidance_x = -(layer_x**2)

    tol = torch.where(x > 0, 0.5, guidance_tol)
    yz_scale = (
        (1 - guidance_x) * tol * ((z**2 + y**2) / ((z / h) ** 2 + (y / w) ** 2)) ** 0.5
    )
    yz_scale.nan_to_num_(nan=1.0)  # caused by z**2 + y**2 == 0
    guidance_yz = torch.where(
        x > 0,
        k_rejection * torch.exp(-0.5 * (y**2 + z**2) / yz_scale),
        (1 - torch.exp(-0.5 * (y**2 + z**2) / yz_scale)),
    )

    guidance = guidance_x * guidance_yz
    reward_guidance = k_guidance * guidance

    # waypoint passing reward
    reward_waypoint = k_waypoint * wp_passing

    # timeout reward
    reward_timeout = k_timeout * timeout

    # normalization
    if enable_normalization:
        reward_progress /= wp_r_sum

    return (
        reward_progress,
        reward_perception,
        reward_cmd,
        reward_collision,
        reward_guidance,
        reward_waypoint,
        reward_timeout,
        dist_to_wp,
        action,
    )
