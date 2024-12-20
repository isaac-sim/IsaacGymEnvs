from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch import pi

from isaacgym import torch_utils
from isaacgym.gymapi import Gym, Env, DOMAIN_SIM
from isaacgymenvs.utils.torch_jit_utils import quaternion_to_matrix
from ..env import OrbitVisData, WallRegionVisData
from ..env.env_creator import EnvCreator, get_gate_actor_name
from ..waypoint import WaypointData


@dataclass
class RandObstacleOptions:
    # extra distance added to waypoints' half diagonal lengths to make safe spheres
    extra_clearance: float = 1.42

    # number of orbital obstacles per surface area m^2 of the orbit spheres
    orbit_density: float = 1 / 6

    # number of trees per meter on the line segments connecting waypoints
    tree_density: float = 1 / 3

    # number of walls per m^2 of the mid-planes
    wall_density: float = 1 / 25

    # a scaling factor for wall center deviation from the line
    wall_dist_scale: float = 1.0

    # scaling factor of standard deviation of obstacles' normal distribution
    std_dev_scale: float = 1.0

    # minimum ground distance to the lowest waypoints' safety sphere
    gnd_distance_min: float = 1.0

    # maximum ground distance to the lowest waypoints' safety sphere
    # final gnd height will not be lower than -backstage_z_offset
    gnd_distance_max: float = 20.0


class ObstacleManager:
    """
    Manages obstacle poses around 4-waypoint tracks.
    """

    def __init__(self, env_creator: EnvCreator):
        assert env_creator.envs_created

        self.gym: Gym = env_creator.gym
        self.envs: List[Env] = env_creator.envs

        self.num_envs = env_creator.params.num_envs
        self.env_size = env_creator.params.env_size
        self.backstage_z_offset = env_creator.params.backstage_z_offset
        self.gate_bar_len_x = env_creator.params.gate_bar_len_x
        self.gate_bar_len_y = env_creator.params.gate_bar_len_y
        self.gate_bar_len_z = env_creator.params.gate_bar_len_z

        self.ground_actor_id = env_creator.ground_actor_id
        self.orbit_actor_id = torch.cat(
            (
                env_creator.box_actor_id,
                env_creator.capsule_actor_id,
                env_creator.cuboid_wireframe_actor_id,
                env_creator.cylinder_actor_id,
                env_creator.hollow_cuboid_actor_id,
                env_creator.sphere_actor_id,
            ),
            dim=1,
        )

        self.tree_actor_id = env_creator.tree_actor_id
        self.wall_actor_id = env_creator.wall_actor_id
        self.gate_bar_actor_id = env_creator.gate_bar_actor_id
        self.obs_no_gnd_actor_flat_id = torch.cat(
            (self.orbit_actor_id, self.tree_actor_id, self.wall_actor_id),
            dim=1,
        ).flatten()
        self.obs_all_actor_flat_id = torch.cat(
            (
                self.ground_actor_id.flatten(),
                self.obs_no_gnd_actor_flat_id,
                self.gate_bar_actor_id.flatten(),
            )
        )

        self.actor_pose = torch.zeros(self.num_envs * env_creator.num_actors_per_env, 7)
        self.vis_data_updated = False

        self.orbit_vis_data: Optional[OrbitVisData] = None
        self.wall_vis_data: Optional[WallRegionVisData] = None

    def compute(
        self, waypoint_data: WaypointData, rand_obs_opts: RandObstacleOptions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a pose tensor holding updated poses for obstacle actors: ground, gate bars,
        orbit obstacles, trees, and walls. The shape of the pose tensor is
        (num_envs * num_actors_per_env, 7), every row is: [px, py, pz, qx, qy, qz, qw].
        Additionally, the flat id of updated actors is returned as 1-dim tensor.
        Also, the data for visualization is updated.

        Args:
            waypoint_data: an instance of ``WaypointData``.
            rand_obs_opts: an instance of ``RandActorOptions``.

        Returns:
            - Actor pose tensor (num_envs * num_actors_per_env, 7).
            - Flat id of updated obstacle actors.
        """

        # only for 4-waypoint envs, and only wp[0] and wp[1] will be given gates
        assert waypoint_data.num_envs == self.num_envs
        assert waypoint_data.num_waypoints == 4

        # refresh the pose tensor, move all actors to backstage
        self.actor_pose[:, :3] = torch.rand_like(
            self.actor_pose[:, :3]
        ) * self.env_size - torch.tensor(
            [
                self.env_size / 2,
                self.env_size / 2,
                self.env_size + self.backstage_z_offset,
            ]
        )
        self.actor_pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        # ========== gate ==========

        # calculate gate bar actors' states
        enable_gate = waypoint_data.gate_flag >= 0.5
        gate_hor_bar_len_y = torch.ceil(waypoint_data.width)
        gate_ver_bar_len_y = torch.ceil(waypoint_data.height)
        q_hor_to_ver = torch_utils.quat_from_euler_xyz(
            torch.tensor(pi / 2), torch.tensor(0.0), torch.tensor(0.0)
        )

        # TODO: whoa! nested loops! what have I done? multiprocessing?
        for env_id in range(self.num_envs):
            for wp_id in [1, 2]:
                for bar_id in range(4):
                    if enable_gate[env_id, wp_id]:
                        wp_p = waypoint_data.position[env_id, wp_id]
                        wp_q = waypoint_data.quaternion[env_id, wp_id]
                        wp_mat = quaternion_to_matrix(wp_q.roll(1))
                        wp_axis_y = wp_mat[:, 1]
                        wp_axis_z = wp_mat[:, 2]

                        bar_len_x = self.gate_bar_len_x[
                            int(waypoint_data.gate_x_len_choice[env_id, wp_id].round())
                        ]
                        bar_len_z = self.gate_bar_len_z[
                            int(waypoint_data.gate_weight_choice[env_id, wp_id].round())
                        ]

                        if bar_id == 0:
                            # 0: horizontal top (z+)
                            bar_len_y = gate_hor_bar_len_y[env_id, wp_id]
                            actor_q = wp_q
                            actor_p = (
                                wp_p
                                + wp_axis_z
                                * (waypoint_data.height[env_id, wp_id] + bar_len_z)
                                / 2
                            )
                        elif bar_id == 1:
                            # 1: horizontal bottom (z-)
                            bar_len_y = gate_hor_bar_len_y[env_id, wp_id]
                            actor_q = wp_q
                            actor_p = (
                                wp_p
                                - wp_axis_z
                                * (waypoint_data.height[env_id, wp_id] + bar_len_z)
                                / 2
                            )
                        elif bar_id == 2:
                            # 2: vertical left (y+)
                            bar_len_y = gate_ver_bar_len_y[env_id, wp_id]
                            actor_q = torch_utils.quat_mul(wp_q, q_hor_to_ver)
                            actor_p = (
                                wp_p
                                + wp_axis_y
                                * (waypoint_data.width[env_id, wp_id] + bar_len_z)
                                / 2
                            )
                        else:
                            # 3: vertical right (y-)
                            bar_len_y = gate_ver_bar_len_y[env_id, wp_id]
                            actor_q = torch_utils.quat_mul(wp_q, q_hor_to_ver)
                            actor_p = (
                                wp_p
                                - wp_axis_y
                                * (waypoint_data.width[env_id, wp_id] + bar_len_z)
                                / 2
                            )

                        actor_name = get_gate_actor_name(
                            wp_id, bar_len_x, float(bar_len_y), bar_len_z, bar_id
                        )
                        gate_actor_id = self.gym.find_actor_index(
                            self.envs[env_id], actor_name, DOMAIN_SIM
                        )
                        self.actor_pose[gate_actor_id, :3] = actor_p
                        self.actor_pose[gate_actor_id, 3:7] = actor_q

        # ========== orbit, trees, walls ==========

        # calculate orbit spheres
        r_wp = (
            waypoint_data.width**2 + waypoint_data.height**2
        ) ** 0.5 + rand_obs_opts.extra_clearance  # (num_envs, num_waypoints)
        r_orbit_min = r_wp[:, :2]  # (num_envs, 2)
        r_orbit_max = waypoint_data.r[:, :2] - r_wp[:, 1:3]
        r_orbit_max.clamp_(min=r_orbit_min)
        r_orbit_range = r_orbit_max - r_orbit_min
        r_orbit_mean = (r_orbit_max + r_orbit_min) / 2

        # calculate total number of actors for each category to put onto stage
        orbit_area = 4 * pi * torch.linalg.norm(r_orbit_mean, dim=1) ** 2
        tree_length = torch.sum(r_orbit_range, dim=1)
        wall_area = torch.linalg.norm(waypoint_data.r[:, :2], dim=1) ** 2

        num_orbit = orbit_area * rand_obs_opts.orbit_density
        num_trees = tree_length * rand_obs_opts.tree_density
        num_walls = wall_area * rand_obs_opts.wall_density

        torch.ceil(input=num_orbit, out=num_orbit)
        torch.ceil(input=num_trees, out=num_trees)
        torch.ceil(input=num_walls, out=num_walls)

        num_orbit_env_total = int(self.orbit_actor_id.shape[1])
        num_trees_env_total = int(self.tree_actor_id.shape[1])
        num_walls_env_total = int(self.wall_actor_id.shape[1])

        num_orbit.clamp_(max=num_orbit_env_total)
        num_trees.clamp_(max=num_trees_env_total)
        num_walls.clamp_(max=num_walls_env_total)

        # loop through envs to fill in state buffer
        # TODO: this is too slow for large num of envs
        square_vis_q = torch.zeros(self.num_envs, 2, 4)
        for i in range(self.num_envs):
            # allocate actor id
            if self.orbit_actor_id.shape[1] == 0:
                init_orbit_actor_id = 0
            else:
                init_orbit_actor_id = int(self.orbit_actor_id[i, 0])
            num_orbit_list, orbit_actor_id_list = allocate_actor_id(
                int(num_orbit[i]),
                num_orbit_env_total,
                init_orbit_actor_id,
                float(r_orbit_mean[i, 0] ** 2),
                float(r_orbit_mean[i, 1] ** 2),
            )

            if self.tree_actor_id.shape[1] == 0:
                init_tree_actor_id = 0
            else:
                init_tree_actor_id = int(self.tree_actor_id[i, 0])
            num_trees_list, tree_actor_id_list = allocate_actor_id(
                int(num_trees[i]),
                num_trees_env_total,
                init_tree_actor_id,
                float(r_orbit_range[i, 0]),
                float(r_orbit_range[i, 1]),
            )

            if self.wall_actor_id.shape[1] == 0:
                init_wall_actor_id = 0
            else:
                init_wall_actor_id = int(self.wall_actor_id[i, 0])
            num_walls_list, wall_actor_id_list = allocate_actor_id(
                int(num_walls[i]),
                num_walls_env_total,
                init_wall_actor_id,
                float(waypoint_data.r[i, 0] ** 2),
                float(waypoint_data.r[i, 1] ** 2),
            )

            # sample and calculate actors' state
            for j in range(2):
                # orbit
                psi_rand = torch.rand(num_orbit_list[j]) * 2 * pi
                theta_rand = torch.arcsin(2 * torch.rand(num_orbit_list[j]) - 1)
                r_std_dev = r_orbit_range[i, j] / 6 * rand_obs_opts.std_dev_scale
                r_rand = torch.randn(num_orbit_list[j]) * r_std_dev + r_orbit_mean[i, j]
                q_rand = torch.rand(num_orbit_list[j], 4)
                q_rand /= torch.linalg.norm(q_rand, dim=1, keepdim=True)

                self.actor_pose[orbit_actor_id_list[j], 0] = (
                    r_rand * torch.cos(theta_rand) * torch.cos(psi_rand)
                    + waypoint_data.position[i, j, 0]
                )
                self.actor_pose[orbit_actor_id_list[j], 1] = (
                    r_rand * torch.cos(theta_rand) * torch.sin(psi_rand)
                    + waypoint_data.position[i, j, 1]
                )
                self.actor_pose[orbit_actor_id_list[j], 2] = (
                    r_rand * torch.sin(theta_rand) + waypoint_data.position[i, j, 2]
                )
                self.actor_pose[orbit_actor_id_list[j], 3:7] = q_rand

                # tree
                x_rand = (
                    torch.rand(num_trees_list[j]) * r_orbit_range[i, j]
                    + r_orbit_min[i, j]
                )
                roll_rand = torch.rand(num_trees_list[j]) * 2 * pi

                vec_x = waypoint_data.position[i, j + 1] - waypoint_data.position[i, j]
                self.actor_pose[tree_actor_id_list[j], :3] = (
                    x_rand.unsqueeze(1) * vec_x / torch.linalg.norm(vec_x)
                    + waypoint_data.position[i, j]
                )

                q = torch_utils.quat_mul(
                    waypoint_data.quaternion[i, j],
                    torch_utils.quat_from_euler_xyz(
                        torch.zeros_like(waypoint_data.theta[i, j]),
                        -waypoint_data.theta[i, j],
                        waypoint_data.psi[i, j],
                    ),
                )
                local_coord_q = torch.zeros(num_trees_list[j], 4)
                local_coord_q[:] = q
                rot_q_rand = torch_utils.quat_from_euler_xyz(
                    roll_rand,
                    torch.zeros_like(roll_rand),
                    torch.zeros_like(roll_rand),
                )
                self.actor_pose[tree_actor_id_list[j], 3:7] = torch_utils.quat_mul(
                    local_coord_q, rot_q_rand
                )

                # wall
                yz_rand = (
                    torch.rand(num_walls_list[j], 2) * waypoint_data.r[i, j]
                    - waypoint_data.r[i, j] / 2
                ) * rand_obs_opts.wall_dist_scale
                x_rand = (
                    torch.randn(num_walls_list[j]) * waypoint_data.r[i, j] / 6
                    + waypoint_data.r[i, j] / 2
                )

                self.actor_pose[wall_actor_id_list[j], 3:7] = q
                local_coord_mat = quaternion_to_matrix(q.roll(1))
                vec_x = local_coord_mat[:, 0]
                vec_y = local_coord_mat[:, 1]
                vec_z = local_coord_mat[:, 2]
                self.actor_pose[wall_actor_id_list[j], :3] = (
                    x_rand.unsqueeze(1) * vec_x
                    + yz_rand[:, 0].unsqueeze(1) * vec_y
                    + yz_rand[:, 1].unsqueeze(1) * vec_z
                    + waypoint_data.position[i, j]
                )
                square_vis_q[i, j] = q

        # remove obstacles too close to waypoints
        actor_pose = self.actor_pose[self.obs_no_gnd_actor_flat_id].view(
            self.num_envs, -1, 7
        )
        actor_pos = actor_pose[:, :, :3]
        wp_pos_check = waypoint_data.position[:, :3]
        dist = torch.linalg.norm(
            actor_pos.unsqueeze(2) - wp_pos_check.unsqueeze(1), dim=-1
        )
        wp_safe_dist: torch.Tensor = r_wp[:, :3]
        too_close = (dist < wp_safe_dist.unsqueeze(1)).any(dim=-1)
        actor_pose[:, :, 2] = torch.where(
            too_close, -self.env_size - self.backstage_z_offset, actor_pose[:, :, 2]
        )

        self.actor_pose[self.obs_no_gnd_actor_flat_id] = actor_pose.view(-1, 7)

        # ========== ground ==========

        wp_pos_z = waypoint_data.position[:, :, -1]
        wp_pos_z_safe = wp_pos_z - r_wp
        wp_pos_z_safe_min = torch.min(wp_pos_z_safe, dim=-1).values
        gnd_z_max = wp_pos_z_safe_min - rand_obs_opts.gnd_distance_min  # (num_envs, )
        gnd_z_min = wp_pos_z_safe_min - rand_obs_opts.gnd_distance_max
        gnd_z_range = gnd_z_max - gnd_z_min
        gnd_z_rand = torch.rand(self.num_envs) * gnd_z_range + gnd_z_min
        gnd_z_rand.clamp_(min=-self.backstage_z_offset)
        self.actor_pose[self.ground_actor_id.flatten(), :2] = 0
        self.actor_pose[self.ground_actor_id.flatten(), 2] = gnd_z_rand

        # ========== visualization ==========

        # create visualization data
        self.orbit_vis_data = OrbitVisData(
            position=waypoint_data.position[:, :2],
            r_min=r_orbit_min,
            r_mean=r_orbit_mean,
            r_max=r_orbit_max,
        )
        self.wall_vis_data = WallRegionVisData(
            position=(waypoint_data.position[:, :2] + waypoint_data.position[:, 1:3])
            / 2,
            quaternion=square_vis_q,
            dim=waypoint_data.r[:, :2],
        )
        self.vis_data_updated = True

        return self.actor_pose, self.obs_all_actor_flat_id

    def get_vis_data(self) -> Tuple[OrbitVisData, WallRegionVisData]:
        assert self.vis_data_updated is True
        return self.orbit_vis_data, self.wall_vis_data


def allocate_actor_id(
    num_to_alloc: int,
    num_env_total: int,
    init_actor_id: int,
    portion_0: float,
    portion_1: float,
) -> Tuple[List[int], List[torch.Tensor]]:
    if portion_0 + portion_1 == 0:
        num_0 = num_1 = 0
    else:
        num_0 = min(
            num_to_alloc, int(num_to_alloc * portion_0 / (portion_0 + portion_1))
        )
        num_1 = num_to_alloc - num_0

    permuted_actor_id = torch.randperm(num_env_total) + init_actor_id
    alloc_actor_id = [
        permuted_actor_id[:num_0],
        permuted_actor_id[num_0:num_to_alloc],
    ]

    return [num_0, num_1], alloc_actor_id
