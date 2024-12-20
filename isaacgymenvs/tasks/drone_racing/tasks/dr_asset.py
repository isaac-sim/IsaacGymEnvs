from typing import Dict, Any, Optional, List

import torch

from isaacgym import gymapi
from isaacgym.gymapi import Vec3, Asset, Transform, AssetOptions
from isaacgymenvs.tasks.drone_racing.assets import (
    create_drone_quadcopter,
    DroneQuadcopterOptions,
    create_track_rmua,
    create_track_splits,
    create_track_walls,
    create_track_multistory,
    create_track_geom_kebab,
    create_track_planar_circle,
    create_track_wavy_eight,
    create_track_turns,
    create_track_simple_stick,
    TrackMultiStoryOptions,
    TrackSplitsOptions,
    TrackWallsOptions,
    TrackRmuaOptions,
    TrackGeomKebabOptions,
    TrackPlanarCircleOptions,
    TrackWavyEightOptions,
    TrackTurnsOptions,
    TrackSimpleStickOptions,
)
from isaacgymenvs.tasks.drone_racing.waypoint import (
    WaypointData,
    Waypoint,
)
from .dr_default_out import DRDefaultOut


class DRAsset(DRDefaultOut):

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
        self.asset_name = cfg["assetName"]
        is_asset_name_valid = (
            self.asset_name == "multistory"
            or self.asset_name == "rmua"
            or self.asset_name == "splits"
            or self.asset_name == "walls"
            or self.asset_name == "geom_kebab"
            or self.asset_name == "geom_kebab_no_obst"
            or self.asset_name == "planar_circle"
            or self.asset_name == "planar_circle_no_obst"
            or self.asset_name == "wavy_eight"
            or self.asset_name == "wavy_eight_no_obst"
            or self.asset_name == "turns"
            or self.asset_name == "simple_stick"
            or self.asset_name == "simple_stick_no_obst"
        )
        assert is_asset_name_valid
        self.gnd_offset: float = cfg["env"]["groundOffset"]
        self.disable_gnd: bool = cfg["env"]["disableGround"]
        self.appended_wp_dist: float = cfg["env"]["appendWpDist"]

        self.track_wp_data: Optional[WaypointData] = None
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        assert self.track_wp_data is not None
        self.waypoint_data = self.track_wp_data
        if self.viewer and self.enable_debug_viz:
            self.waypoint_data.visualize(self.gym, self.envs, self.viewer, 1)

    def _create_envs(self):
        # create track asset
        track_asset: Optional[Asset] = None
        track_wp_list: List[Waypoint] = []

        if self.asset_name == "multistory":
            track_asset, track_wp_list = create_track_multistory(
                self.gym, self.sim, TrackMultiStoryOptions()
            )
        elif self.asset_name == "rmua":
            track_asset, track_wp_list = create_track_rmua(
                self.gym, self.sim, TrackRmuaOptions()
            )
        elif self.asset_name == "splits":
            track_asset, track_wp_list = create_track_splits(
                self.gym, self.sim, TrackSplitsOptions()
            )
        elif self.asset_name == "walls":
            track_asset, track_wp_list = create_track_walls(
                self.gym, self.sim, TrackWallsOptions()
            )
        elif self.asset_name == "geom_kebab":
            track_asset, track_wp_list = create_track_geom_kebab(
                self.gym, self.sim, TrackGeomKebabOptions(add_obstacles=True)
            )
        elif self.asset_name == "geom_kebab_no_obst":
            track_asset, track_wp_list = create_track_geom_kebab(
                self.gym, self.sim, TrackGeomKebabOptions(add_obstacles=False)
            )
        elif self.asset_name == "planar_circle":
            track_asset, track_wp_list = create_track_planar_circle(
                self.gym, self.sim, TrackPlanarCircleOptions(add_obstacles=True)
            )
        elif self.asset_name == "planar_circle_no_obst":
            track_asset, track_wp_list = create_track_planar_circle(
                self.gym, self.sim, TrackPlanarCircleOptions(add_obstacles=False)
            )
        elif self.asset_name == "wavy_eight":
            track_asset, track_wp_list = create_track_wavy_eight(
                self.gym, self.sim, TrackWavyEightOptions(add_obstacles=True)
            )
        elif self.asset_name == "wavy_eight_no_obst":
            track_asset, track_wp_list = create_track_wavy_eight(
                self.gym, self.sim, TrackWavyEightOptions(add_obstacles=False)
            )
        elif self.asset_name == "turns":
            track_asset, track_wp_list = create_track_turns(
                self.gym, self.sim, TrackTurnsOptions()
            )
        elif self.asset_name == "simple_stick":
            track_asset, track_wp_list = create_track_simple_stick(
                self.gym, self.sim, TrackSimpleStickOptions()
            )
        elif self.asset_name == "simple_stick":
            track_asset, track_wp_list = create_track_simple_stick(
                self.gym, self.sim, TrackSimpleStickOptions(add_obstacles=True)
            )
        elif self.asset_name == "simple_stick_no_obst":
            track_asset, track_wp_list = create_track_simple_stick(
                self.gym, self.sim, TrackSimpleStickOptions(add_obstacles=False)
            )

        self.track_wp_data = WaypointData.from_waypoint_list(
            self.num_envs, track_wp_list, True, self.appended_wp_dist
        )

        # drone asset
        drone_asset = create_drone_quadcopter(
            self.gym,
            self.sim,
            self._param_from_cfg(
                DroneQuadcopterOptions, self.cfg["droneSim"]["drone_asset_options"]
            ),
        )

        # ground asset
        static_asset_opts: AssetOptions = AssetOptions()
        static_asset_opts.fix_base_link = True
        static_asset_opts.disable_gravity = True
        static_asset_opts.collapse_fixed_joints = True
        ground_asset = self.gym.create_box(self.sim, 40, 40, 0.3, static_asset_opts)

        # create envs
        tf = Transform()
        for i in range(self.num_envs):
            env = self.gym.create_env(
                self.sim, Vec3(-20, -20, 0), Vec3(20, 20, 40), int(self.num_envs**0.5)
            )
            self.envs.append(env)

            # create drone
            drone_actor = self.gym.create_actor(env, drone_asset, tf, "drone", i, 0)
            self.drone_actors.append(drone_actor)

            # create track
            self.gym.create_actor(env, track_asset, tf, "track", i, 1)

            # create ground
            if not self.disable_gnd:
                tf_gnd = Transform()
                tf_gnd.p.z = -0.15 + self.gnd_offset
                ground_actor = self.gym.create_actor(
                    env, ground_asset, tf_gnd, "ground", i, 1
                )
                self.gym.set_rigid_body_color(
                    env,
                    ground_actor,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.25, 0.25, 0.25),
                )

        if not self.disable_gnd:
            self.num_actors_per_env = 3
        else:
            self.num_actors_per_env = 2

        self.drone_actor_id_flat = torch.arange(
            0,
            self.num_envs * self.num_actors_per_env,
            step=self.num_actors_per_env,
            device=self.device,
        )
        self.num_waypoints_to_track = self.track_wp_data.num_waypoints - 1
        self.env_size = 40

    def _randomize_racing_tracks(self):
        pass
