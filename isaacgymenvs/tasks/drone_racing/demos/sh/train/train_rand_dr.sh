if [ "$#" -ne 2 ] && [ "$#" -ne 3 ]; then
  echo "Error: incorrect arg count"
  echo "Require: wandb max_iter [checkpoint]"
  echo "Example: ./train_rand_dr.sh True 150"
  exit 1
fi

cd ../../../../../

python train.py task=DRRandom \
  headless=False \
  wandb_activate=$1 \
  max_iterations=$2 \
  num_envs=512 \
  task.droneSim.drone_asset_options.disable_visuals=False \
  task.env.enableDebugVis=False \
  task.env.logging.enable=False \
  task.env.maxEpisodeLength=150 \
  task.env.enableStrictCollision=True \
  task.envCreator.num_box_actors=0 \
  task.envCreator.num_box_assets=0 \
  task.envCreator.num_capsule_actors=0 \
  task.envCreator.num_capsule_assets=0 \
  task.envCreator.num_cuboid_wireframe_actors=0 \
  task.envCreator.num_cuboid_wireframe_assets=0 \
  task.envCreator.num_cylinder_actors=0 \
  task.envCreator.num_cylinder_assets=0 \
  task.envCreator.num_hollow_cuboid_actors=0 \
  task.envCreator.num_hollow_cuboid_assets=0 \
  task.envCreator.num_sphere_actors=0 \
  task.envCreator.num_sphere_assets=0 \
  task.envCreator.num_tree_actors=4 \
  task.envCreator.num_tree_assets=4 \
  task.envCreator.num_wall_actors=12 \
  task.envCreator.num_wall_assets=12 \
  task.initRandOpt.randDroneOptions.dist_along_line_max=0.25 \
  task.initRandOpt.randDroneOptions.drone_rotation_x_max=1 \
  task.initRandOpt.randDroneOptions.dist_to_line_max=2.0 \
  task.initRandOpt.randDroneOptions.lin_vel_x_max=1 \
  task.initRandOpt.randDroneOptions.lin_vel_y_max=1 \
  task.initRandOpt.randDroneOptions.lin_vel_z_max=1 \
  task.initRandOpt.randDroneOptions.ang_vel_x_max=1 \
  task.initRandOpt.randDroneOptions.ang_vel_y_max=1 \
  task.initRandOpt.randDroneOptions.ang_vel_z_max=1 \
  task.initRandOpt.randDroneOptions.aileron_max=0.2 \
  task.initRandOpt.randDroneOptions.elevator_max=0.2 \
  task.initRandOpt.randDroneOptions.rudder_max=0.2 \
  task.initRandOpt.randDroneOptions.throttle_min=-1 \
  task.initRandOpt.randDroneOptions.throttle_max=-0.5 \
  task.initRandOpt.randCameraOptions.d_x_max=0.01 \
  task.initRandOpt.randCameraOptions.d_y_max=0 \
  task.initRandOpt.randCameraOptions.d_z_max=0.01 \
  task.initRandOpt.randCameraOptions.d_angle_max=5 \
  task.initRandOpt.randWaypointOptions.wp_size_min=1.4 \
  task.initRandOpt.randWaypointOptions.wp_size_max=2.0 \
  task.initRandOpt.randWaypointOptions.init_roll_max=0.2 \
  task.initRandOpt.randWaypointOptions.init_pitch_max=0.2 \
  task.initRandOpt.randWaypointOptions.init_yaw_max=3.14 \
  task.initRandOpt.randWaypointOptions.psi_max=0.3 \
  task.initRandOpt.randWaypointOptions.theta_max=0.3 \
  task.initRandOpt.randWaypointOptions.alpha_max=3.14 \
  task.initRandOpt.randWaypointOptions.gamma_max=0.2 \
  task.initRandOpt.randWaypointOptions.r_min=6 \
  task.initRandOpt.randWaypointOptions.r_max=18 \
  task.initRandOpt.randWaypointOptions.force_gate_flag=-1 \
  task.initRandOpt.randWaypointOptions.same_track=0 \
  task.initRandOpt.randObstacleOptions.extra_clearance=1.4 \
  task.initRandOpt.randObstacleOptions.orbit_density=0 \
  task.initRandOpt.randObstacleOptions.tree_density=1 \
  task.initRandOpt.randObstacleOptions.wall_density=1 \
  task.initRandOpt.randObstacleOptions.wall_dist_scale=0.67 \
  task.initRandOpt.randObstacleOptions.std_dev_scale=1 \
  task.initRandOpt.randObstacleOptions.gnd_distance_min=2 \
  task.initRandOpt.randObstacleOptions.gnd_distance_max=5 \
  train.params.algo.name=dr_continuous \
  train.params.config.horizon_length=1024 \
  train.params.config.minibatch_size=32768
