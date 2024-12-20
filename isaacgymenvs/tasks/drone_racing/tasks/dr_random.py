from typing import Dict, Any, Optional

from isaacgymenvs.tasks.drone_racing.env import (
    EnvCreatorParams,
    EnvCreator,
)
from isaacgymenvs.tasks.drone_racing.managers import (
    ObstacleManager,
    RandObstacleOptions,
)
from isaacgymenvs.tasks.drone_racing.waypoint import (
    WaypointGeneratorParams,
    WaypointGenerator,
    RandWaypointOptions,
)
from .dr_default_out import DRDefaultOut


class DRRandom(DRDefaultOut):
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
        self.env_creator: Optional[EnvCreator] = None
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        # extra modules to generate random multiple tracks with or without obstacles
        self.waypoint_generator = WaypointGenerator(
            self._param_from_cfg(WaypointGeneratorParams, self.cfg["waypointGenerator"])
        )
        self.obstacle_manager = ObstacleManager(self.env_creator)
        self.disable_obstacle_man = self.cfg["disableObstacleManager"]

        # extra random generator options
        self.rand_waypoint_opts = self._param_from_cfg(
            RandWaypointOptions, self.cfg["initRandOpt"]["randWaypointOptions"]
        )
        self.rand_obstacle_opts = self._param_from_cfg(
            RandObstacleOptions, self.cfg["initRandOpt"]["randObstacleOptions"]
        )

    def set_train_info(self, env_frames, *args, **kwargs):
        rand_waypoint_opts = kwargs.get("rand_waypoint_opts", None)
        rand_obstacle_opts = kwargs.get("rand_obstacle_opts", None)
        rand_drone_opts = kwargs.get("rand_drone_opts", None)
        rand_camera_opts = kwargs.get("rand_camera_opts", None)

        if rand_waypoint_opts is not None:
            self.rand_waypoint_opts = rand_waypoint_opts
        if rand_obstacle_opts is not None:
            self.rand_obstacle_opts = rand_obstacle_opts
        if rand_drone_opts is not None:
            self.rand_drone_opts = rand_drone_opts
        if rand_camera_opts is not None:
            self.rand_camera_opts = rand_camera_opts

    def _create_envs(self):
        # create envs
        self.env_creator = EnvCreator(
            self.gym, self.sim, self._get_env_creator_params()
        )
        self.env_creator.create([0.0, 0.0, self.env_creator.params.env_size / 2])

        # assign required variables
        self.num_actors_per_env = self.env_creator.num_actors_per_env
        self.drone_actor_id_flat = self.env_creator.drone_actor_id.flatten().to(
            device=self.device
        )
        self.envs = self.env_creator.envs
        self.drone_actors = self.env_creator.quad_actors
        self.num_waypoints_to_track = self.cfg["waypointGenerator"]["num_waypoints"] - 1
        self.env_size = self.env_creator.params.env_size

    def _randomize_racing_tracks(self):
        # generate random waypoints for multiple tracks
        self.waypoint_data = self.waypoint_generator.compute(self.rand_waypoint_opts)
        if self.viewer and self.enable_debug_viz:
            self.gym.clear_lines(self.viewer)
            self.waypoint_data.visualize(self.gym, self.envs, self.viewer, 1)

        # place random obstacles around the waypoints if enabled
        # sometimes we do not want to compute gate and obstacles at all
        # e.g. for large amount of envs, state-only drone racing
        if not self.disable_obstacle_man:
            obs_actor_pose, obs_actor_id = self.obstacle_manager.compute(
                waypoint_data=self.waypoint_data, rand_obs_opts=self.rand_obstacle_opts
            )
            self.actor_root_state[obs_actor_id, :7] = obs_actor_pose[obs_actor_id].to(
                self.device
            )

    def _get_env_creator_params(self) -> EnvCreatorParams:
        p = EnvCreatorParams()
        for opt in self.cfg["envCreator"].keys():
            if opt == "drone_asset_options":
                for drone_asset_opt in self.cfg["envCreator"][opt].keys():
                    assert hasattr(p.drone_asset_options, drone_asset_opt)
                    setattr(
                        p.drone_asset_options,
                        drone_asset_opt,
                        self.cfg["envCreator"][opt][drone_asset_opt],
                    )
            else:
                assert hasattr(p, opt), opt
                setattr(p, opt, self.cfg["envCreator"][opt])
        p.num_envs = self.num_envs
        p.device = self.device
        return p
