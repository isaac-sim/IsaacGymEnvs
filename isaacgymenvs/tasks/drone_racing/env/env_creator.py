from dataclasses import dataclass, field
from typing import List

import torch
from tqdm import tqdm

from isaacgym import gymapi
from isaacgym.gymapi import Gym, Sim, Vec3, Env, Asset, AssetOptions
from isaacgym.gymapi import Transform, MESH_VISUAL
from ..assets import (
    DroneQuadcopterOptions,
    create_drone_quadcopter,
    CollectionBox,
    CollectionCapsule,
    CollectionCuboidWireframe,
    CollectionCylinder,
    CollectionHollowCuboid,
    CollectionSphere,
    CollectionTree,
    CollectionBoxOptions,
    CollectionCapsuleOptions,
    CollectionCuboidWireframeOptions,
    CollectionCylinderOptions,
    CollectionHollowCuboidOptions,
    CollectionSphereOptions,
    CollectionTreeOptions,
)


@dataclass
class EnvCreatorParams:
    # number of environments to create and manage
    num_envs: int = 64

    # size of the environment bounding box [m]
    env_size: float = 40.0

    # positive offset to move obstacles further down [m]
    backstage_z_offset: float = 20.0

    # ground color
    ground_color: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25])

    # ground length in z direction (thickness) [m]
    ground_len_z: float = 0.3

    # all possible length x, ascending order
    gate_bar_len_x: List[float] = field(default_factory=lambda: [0.15, 0.3])

    # all possible length y, ascending order
    gate_bar_len_y: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])

    # all possible length z, ascending order
    gate_bar_len_z: List[float] = field(default_factory=lambda: [0.15, 0.3])

    # gate color in rgb [0, 1]
    gate_color: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.3])

    # enabling tqdm shows progress of loading assets and creating envs
    disable_tqdm: bool = False

    # drone asset options, here we assume quadcopter
    drone_asset_options: DroneQuadcopterOptions = DroneQuadcopterOptions()

    # other asset options
    static_asset_opts: AssetOptions = AssetOptions()
    static_asset_opts.fix_base_link = True
    static_asset_opts.disable_gravity = True
    static_asset_opts.collapse_fixed_joints = True

    # random boxes, params: [size_x, size_y, size_z]
    num_box_actors: int = 20  # per env
    num_box_assets: int = 100
    box_params_min: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2])
    box_params_max: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
    box_color: List[float] = field(
        default_factory=lambda: [31 / 255, 119 / 255, 180 / 255]
    )

    # random capsules, params: [radius, length]
    num_capsule_actors: int = 20  # per env
    num_capsule_assets: int = 100
    capsule_params_min: List[float] = field(default_factory=lambda: [0.1, 0.1])
    capsule_params_max: List[float] = field(default_factory=lambda: [1.0, 1.0])
    capsule_color: List[float] = field(
        default_factory=lambda: [186 / 255, 54 / 255, 87 / 255]
    )

    # random cuboid wireframes, params: [size_x, size_y, size_z, weight]
    num_cuboid_wireframe_actors: int = 20  # per env
    num_cuboid_wireframe_assets: int = 100
    cuboid_wireframe_params_min: List[float] = field(
        default_factory=lambda: [0.2, 0.2, 0.2, 0.1]
    )
    cuboid_wireframe_params_max: List[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 0.4]
    )
    cuboid_wireframe_color: List[float] = field(
        default_factory=lambda: [148 / 255, 103 / 255, 189 / 255]
    )

    # random cylinders, params: [radius, length]
    num_cylinder_actors: int = 20  # per env
    num_cylinder_assets: int = 100
    cylinder_params_min: List[float] = field(default_factory=lambda: [0.1, 0.2])
    cylinder_params_max: List[float] = field(default_factory=lambda: [1.0, 2.0])
    cylinder_color: List[float] = field(
        default_factory=lambda: [140 / 255, 86 / 255, 75 / 255]
    )

    # random hollow cuboids, params: [length_x, inner_length_y, inner_length_z, diff_length_y, diff_length_z]
    num_hollow_cuboid_actors: int = 20  # per env
    num_hollow_cuboid_assets: int = 100
    hollow_cuboid_params_min: List[float] = field(
        default_factory=lambda: [0.05, 0.5, 0.5, 0.2, 0.2]
    )
    hollow_cuboid_params_max: List[float] = field(
        default_factory=lambda: [0.25, 1.4, 1.4, 0.6, 0.6]
    )
    hollow_cuboid_color: List[float] = field(
        default_factory=lambda: [227 / 255, 119 / 255, 194 / 255]
    )

    # random spheres, params: [radius]
    num_sphere_actors: int = 20  # per env
    num_sphere_assets: int = 100
    sphere_params_min: List[float] = field(default_factory=lambda: [0.1])
    sphere_params_max: List[float] = field(default_factory=lambda: [1.0])
    sphere_color: List[float] = field(
        default_factory=lambda: [188 / 255, 189 / 255, 34 / 255]
    )

    # random trees, params: none
    num_tree_actors: int = 20  # per env
    num_tree_assets: int = 100
    tree_color: List[float] = field(
        default_factory=lambda: [107 / 255, 138 / 255, 122 / 255]
    )

    # random walls, params: [size_x, size_y, size_z]
    num_wall_actors: int = 20  # per env
    num_wall_assets: int = 100
    wall_params_min: List[float] = field(default_factory=lambda: [0.2, 2.0, 2.0])
    wall_params_max: List[float] = field(default_factory=lambda: [0.2, 4.0, 4.0])
    wall_color: List[float] = field(
        default_factory=lambda: [23 / 255, 190 / 255, 207 / 255]
    )


class EnvCreator:
    """
    Creates drone and obstacle actors for 4-waypoint racing envs.
    """

    def __init__(self, gym: Gym, sim: Sim, params: EnvCreatorParams):
        self.gym: Gym = gym
        self.sim: Sim = sim
        self.params = params

        # ========== create assets ==========

        # drone
        self.drone_asset = create_drone_quadcopter(
            self.gym, self.sim, params.drone_asset_options
        )

        # ground
        ground_x = ground_y = params.env_size
        self.ground_asset = self.gym.create_box(
            self.sim, ground_x, ground_y, params.ground_len_z, params.static_asset_opts
        )

        # gate bars
        self.gate_bar_assets: List[List[List[Asset]]] = []
        for x in params.gate_bar_len_x:
            list_y = []
            for y in params.gate_bar_len_y:
                list_z = []
                for z in params.gate_bar_len_z:
                    bar = self.gym.create_box(
                        self.sim, x, y, z, params.static_asset_opts
                    )
                    list_z.append(bar)
                list_y.append(list_z)
            self.gate_bar_assets.append(list_y)

        # boxes
        self.collection_box = CollectionBox(
            self.gym,
            self.sim,
            CollectionBoxOptions(
                num_envs=params.num_envs,
                num_assets=params.num_box_actors,
                num_blueprints=params.num_box_assets,
                asset_options=params.static_asset_opts,
                disable_tqdm=(params.disable_tqdm or params.num_box_assets == 0),
                params_min=params.box_params_min,
                params_max=params.box_params_max,
            ),
        )

        # capsules
        self.collection_capsule = CollectionCapsule(
            self.gym,
            self.sim,
            CollectionCapsuleOptions(
                num_envs=params.num_envs,
                num_assets=params.num_capsule_actors,
                num_blueprints=params.num_capsule_assets,
                asset_options=params.static_asset_opts,
                disable_tqdm=(params.disable_tqdm or params.num_capsule_assets == 0),
                params_min=params.capsule_params_min,
                params_max=params.capsule_params_max,
            ),
        )

        # cuboid wireframes
        self.collection_cuboid_wireframe = CollectionCuboidWireframe(
            self.gym,
            self.sim,
            CollectionCuboidWireframeOptions(
                num_envs=params.num_envs,
                num_assets=params.num_cuboid_wireframe_actors,
                num_blueprints=params.num_cuboid_wireframe_assets,
                asset_options=params.static_asset_opts,
                disable_tqdm=(
                    params.disable_tqdm or params.num_cuboid_wireframe_assets == 0
                ),
                params_min=params.cuboid_wireframe_params_min,
                params_max=params.cuboid_wireframe_params_max,
            ),
        )

        # cylinders
        self.collection_cylinder = CollectionCylinder(
            self.gym,
            self.sim,
            CollectionCylinderOptions(
                num_envs=params.num_envs,
                num_assets=params.num_cylinder_actors,
                num_blueprints=params.num_cylinder_assets,
                asset_options=params.static_asset_opts,
                disable_tqdm=(params.disable_tqdm or params.num_cylinder_assets == 0),
                params_min=params.cylinder_params_min,
                params_max=params.cylinder_params_max,
            ),
        )

        # hollow cuboids
        self.collection_hollow_cuboid = CollectionHollowCuboid(
            self.gym,
            self.sim,
            CollectionHollowCuboidOptions(
                num_envs=params.num_envs,
                num_assets=params.num_hollow_cuboid_actors,
                num_blueprints=params.num_hollow_cuboid_assets,
                asset_options=params.static_asset_opts,
                disable_tqdm=(
                    params.disable_tqdm or params.num_hollow_cuboid_assets == 0
                ),
                params_min=params.hollow_cuboid_params_min,
                params_max=params.hollow_cuboid_params_max,
            ),
        )

        # spheres
        self.collection_sphere = CollectionSphere(
            self.gym,
            self.sim,
            CollectionSphereOptions(
                num_envs=params.num_envs,
                num_assets=params.num_sphere_actors,
                num_blueprints=params.num_sphere_assets,
                asset_options=params.static_asset_opts,
                disable_tqdm=(params.disable_tqdm or params.num_sphere_assets == 0),
                params_min=params.sphere_params_min,
                params_max=params.sphere_params_max,
            ),
        )

        # trees
        self.collection_tree = CollectionTree(
            self.gym,
            self.sim,
            CollectionTreeOptions(
                num_envs=params.num_envs,
                num_assets=params.num_tree_actors,
                num_blueprints=params.num_tree_assets,
                asset_options=params.static_asset_opts,
                disable_tqdm=(params.disable_tqdm or params.num_tree_assets == 0),
            ),
        )

        # walls
        self.collection_wall = CollectionBox(
            self.gym,
            self.sim,
            CollectionBoxOptions(
                num_envs=params.num_envs,
                num_assets=params.num_wall_actors,
                num_blueprints=params.num_wall_assets,
                asset_options=params.static_asset_opts,
                disable_tqdm=(params.disable_tqdm or params.num_wall_assets == 0),
                params_min=params.wall_params_min,
                params_max=params.wall_params_max,
            ),
        )

        # ========== prepare other variables ==========

        self.envs: List[Env] = []
        self.quad_actors: List[int] = []

        num_gate_bar_actors = (
            8  # 2 (waypoints) * 4 (bars per waypoint)
            * len(params.gate_bar_len_x)
            * len(params.gate_bar_len_y)
            * len(params.gate_bar_len_z)
        )

        self.drone_actor_id = torch.zeros(params.num_envs, 1, dtype=torch.int)
        self.ground_actor_id = torch.zeros(params.num_envs, 1, dtype=torch.int)
        self.gate_bar_actor_id = torch.zeros(
            params.num_envs, num_gate_bar_actors, dtype=torch.int
        )
        self.box_actor_id = torch.zeros(
            params.num_envs, params.num_box_actors, dtype=torch.int
        )
        self.capsule_actor_id = torch.zeros(
            params.num_envs, params.num_capsule_actors, dtype=torch.int
        )
        self.cuboid_wireframe_actor_id = torch.zeros(
            params.num_envs,
            params.num_cuboid_wireframe_actors,
            dtype=torch.int,
        )
        self.cylinder_actor_id = torch.zeros(
            params.num_envs, params.num_cylinder_actors, dtype=torch.int
        )
        self.hollow_cuboid_actor_id = torch.zeros(
            params.num_envs, params.num_hollow_cuboid_actors, dtype=torch.int
        )
        self.sphere_actor_id = torch.zeros(
            params.num_envs, params.num_sphere_actors, dtype=torch.int
        )
        self.tree_actor_id = torch.zeros(
            params.num_envs, params.num_tree_actors, dtype=torch.int
        )
        self.wall_actor_id = torch.zeros(
            params.num_envs, params.num_wall_actors, dtype=torch.int
        )

        self.num_actors_per_env = (
            2
            + num_gate_bar_actors
            + params.num_box_actors
            + params.num_capsule_actors
            + params.num_cuboid_wireframe_actors
            + params.num_cylinder_actors
            + params.num_hollow_cuboid_actors
            + params.num_sphere_actors
            + params.num_tree_actors
            + params.num_wall_actors
        )
        self.envs_created = False

    def create(self, drone_position: List[float]):
        """
        Creates envs, actors.

        Args:
            drone_position: spawning position of the drone.
        """

        count = 0

        for i in tqdm(range(self.params.num_envs), disable=self.params.disable_tqdm):

            # ========== env ==========

            env_size = self.params.env_size
            env = self.gym.create_env(
                self.sim,
                Vec3(-env_size / 2, -env_size / 2, 0),
                Vec3(env_size / 2, env_size / 2, env_size),
                int(self.params.num_envs**0.5),
            )
            self.envs.append(env)

            tf = gymapi.Transform()

            # ========== drone ==========

            x, y, z = drone_position
            tf.p = gymapi.Vec3(x, y, z)
            quad_actor = self.gym.create_actor(env, self.drone_asset, tf, "drone", i, 0)
            self.quad_actors.append(quad_actor)
            self.drone_actor_id[i] = count
            count += 1

            # ========== ground ==========

            tf.p = gymapi.Vec3(0.0, 0.0, -self.params.ground_len_z / 2)
            ground_actor = self.gym.create_actor(
                env, self.ground_asset, tf, "ground", i, 1
            )
            r, g, b = self.params.ground_color
            self.gym.set_rigid_body_color(
                env, ground_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(r, g, b)
            )
            self.ground_actor_id[i] = count
            count += 1

            # ========== gate ==========

            for wp_id in [1, 2]:
                for x_id in range(len(self.params.gate_bar_len_x)):
                    for y_id in range(len(self.params.gate_bar_len_y)):
                        for z_id in range(len(self.params.gate_bar_len_z)):
                            for bar_id in [0, 1, 2, 3]:
                                bar_x = self.params.gate_bar_len_x[x_id]
                                bar_y = self.params.gate_bar_len_y[y_id]
                                bar_z = self.params.gate_bar_len_z[z_id]
                                actor = self.gym.create_actor(
                                    env,
                                    self.gate_bar_assets[x_id][y_id][z_id],
                                    rand_backstage_tf(
                                        self.params.env_size,
                                        self.params.backstage_z_offset,
                                    ),
                                    get_gate_actor_name(
                                        wp_id, bar_x, bar_y, bar_z, bar_id
                                    ),
                                    i,
                                    1,
                                )
                                r, g, b = self.params.gate_color
                                self.gym.set_rigid_body_color(
                                    env,
                                    actor,
                                    0,
                                    gymapi.MESH_VISUAL,
                                    gymapi.Vec3(r, g, b),
                                )
            for j in range(self.gate_bar_actor_id.shape[1]):
                self.gate_bar_actor_id[i, j] = count
                count += 1

            # ========== other obstacles ==========

            collections = [
                self.collection_box,
                self.collection_capsule,
                self.collection_cuboid_wireframe,
                self.collection_cylinder,
                self.collection_hollow_cuboid,
                self.collection_sphere,
                self.collection_tree,
                self.collection_wall,
            ]
            names = [
                "box",
                "capsule",
                "wireframe",
                "cylinder",
                "hollow",
                "sphere",
                "tree",
                "wall",
            ]
            colors = [
                self.params.box_color,
                self.params.capsule_color,
                self.params.cuboid_wireframe_color,
                self.params.cylinder_color,
                self.params.hollow_cuboid_color,
                self.params.sphere_color,
                self.params.tree_color,
                self.params.wall_color,
            ]
            actor_ids = [
                self.box_actor_id,
                self.capsule_actor_id,
                self.cuboid_wireframe_actor_id,
                self.cylinder_actor_id,
                self.hollow_cuboid_actor_id,
                self.sphere_actor_id,
                self.tree_actor_id,
                self.wall_actor_id,
            ]

            for j in range(len(collections)):
                create_collection_actors(
                    gym=self.gym,
                    env=env,
                    env_id=i,
                    env_size=self.params.env_size,
                    z_offset=self.params.backstage_z_offset,
                    assets=collections[j].assets[i],
                    name=names[j],
                    color=colors[j],
                )
                for k in range(actor_ids[j].shape[1]):
                    actor_ids[j][i, k] = count
                    count += 1

        self.envs_created = True


def rand_backstage_tf(env_size: float, z_offset: float) -> Transform:
    x, y, z = (
        torch.rand(3) * torch.tensor(env_size)
        - torch.tensor(
            [
                env_size / 2,
                env_size / 2,
                env_size + z_offset,
            ]
        )
    ).tolist()
    tf = Transform()
    tf.p = Vec3(x, y, z)
    return tf


def create_collection_actors(
    gym: Gym,
    env: Env,
    env_id: int,
    env_size: float,
    z_offset: float,
    assets: List[Asset],
    name: str,
    color: List[float] = None,
):
    if color is None:
        color = [0.5, 0.5, 0.5]
    for i in range(len(assets)):
        actor = gym.create_actor(
            env,
            assets[i],
            rand_backstage_tf(env_size, z_offset),
            name + "_" + str(i),
            env_id,
            1,
        )
        r, g, b = color
        gym.set_rigid_body_color(env, actor, 0, MESH_VISUAL, Vec3(r, g, b))


def get_gate_actor_name(
    wp_id: int, len_x: float, len_y: float, len_z: float, bar_id: int
):
    name = (
        "gate_"
        + str(wp_id)
        + "_("
        + str(len_x)
        + ", "
        + str(len_y)
        + ", "
        + str(len_z)
        + ")"
        + "_"
        + str(bar_id)
    )
    return name
