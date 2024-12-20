# similar to training env in train_rand_dr.sh
# enable all physical gates

if [ "$#" -ne 2 ]; then
    echo "Error: incorrect arg count"
    echo "Require: checkpoint exp_name"
    exit 1
fi

run() {
  python train.py task=DRRandom \
    seed=$1 \
    checkpoint=$2 \
    num_envs=$3 \
    test=True \
    task.droneSim.drone_asset_options.disable_visuals=True \
    task.env.enableDebugVis=False \
    task.env.logging.enable=True \
    task.env.logging.experimentName=$4 \
    task.env.logging.logMainCam=True \
    task.env.logging.logExtraCams=True \
    task.env.logging.maxNumEpisodes=$5 \
    task.env.logging.numStepsPerSave=$6 \
    task.env.maxEpisodeLength=$7 \
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
    task.initRandOpt.randDroneOptions.dist_along_line_max=0.1 \
    task.initRandOpt.randDroneOptions.drone_rotation_x_max=1.57 \
    task.initRandOpt.randDroneOptions.dist_to_line_max=0.0 \
    task.initRandOpt.randDroneOptions.lin_vel_x_max=2 \
    task.initRandOpt.randDroneOptions.lin_vel_y_max=2 \
    task.initRandOpt.randDroneOptions.lin_vel_z_max=2 \
    task.initRandOpt.randDroneOptions.ang_vel_x_max=2 \
    task.initRandOpt.randDroneOptions.ang_vel_y_max=2 \
    task.initRandOpt.randDroneOptions.ang_vel_z_max=2 \
    task.initRandOpt.randDroneOptions.aileron_max=0.5 \
    task.initRandOpt.randDroneOptions.elevator_max=0.5 \
    task.initRandOpt.randDroneOptions.rudder_max=0.5 \
    task.initRandOpt.randDroneOptions.throttle_min=-1 \
    task.initRandOpt.randDroneOptions.throttle_max=0.0 \
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
    task.initRandOpt.randWaypointOptions.force_gate_flag=1 \
    task.initRandOpt.randWaypointOptions.same_track=0 \
    task.initRandOpt.randObstacleOptions.extra_clearance=1.4 \
    task.initRandOpt.randObstacleOptions.orbit_density=0 \
    task.initRandOpt.randObstacleOptions.tree_density=1 \
    task.initRandOpt.randObstacleOptions.wall_density=1 \
    task.initRandOpt.randObstacleOptions.wall_dist_scale=0.67 \
    task.initRandOpt.randObstacleOptions.std_dev_scale=1 \
    task.initRandOpt.randObstacleOptions.gnd_distance_min=2 \
    task.initRandOpt.randObstacleOptions.gnd_distance_max=5
}


cd ../../../../../

# 100 envs, 10 episodes per env
# args: seed, checkpoint, num_envs, exp_name, num_ep, num_steps_save, ep_len
run_out=$(run 0 $1 25 $2 10 50 100000)
run_out=$(run 1 $1 25 $2 10 50 100000)
run_out=$(run 2 $1 25 $2 10 50 100000)
run_out=$(run 3 $1 25 $2 10 50 100000)

ulimit -n 65535

# extract SH_LOG_DIR
log_dir=$(echo "$run_out" | grep 'SH_IO_LOG_DIR:' | cut -d':' -f2- | xargs)
exp_dir=$(dirname $log_dir)

cd tasks/drone_racing/demos/

# process logs, currently it requires logging also images
python process_logs.py \
  --exp_dir $exp_dir \
  --pcd_update_itv 25

# rerun experiment
python rerun_exp.py \
  --exp_dir $exp_dir \
  --vel_max_cmap 20 \
  --traj_line_w 1.5