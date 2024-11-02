if [ "$#" -ne 2 ]; then
    echo "Error: incorrect arg count"
    echo "Require: checkpoint exp_name"
    exit 1
fi

run() {
  python train.py task=DRAsset \
    test=True \
    num_envs=25 \
    task.assetName=turns \
    task.env.appendWpDist=10 \
    task.env.numObservations=120 \
    task.env.obsImgMode=dce \
    task.env.disableGround=True \
    task.env.groundOffset=-10 \
    task.env.enableCameraSensors=True \
    task.env.enableDebugVis=False \
    task.env.maxEpisodeLength=100000 \
    task.env.logging.enable=True \
    task.env.logging.experimentName=$2 \
    task.env.logging.logMainCam=True \
    task.env.logging.logExtraCams=True \
    task.env.logging.maxNumEpisodes=4 \
    task.env.logging.numStepsPerSave=50 \
    task.droneSim.drone_asset_options.disable_visuals=True \
    task.initRandOpt.randDroneOptions.dist_along_line_max=0.1 \
    task.initRandOpt.randDroneOptions.drone_rotation_x_max=1.57 \
    task.initRandOpt.randDroneOptions.dist_to_line_max=0.0 \
    task.initRandOpt.randDroneOptions.lin_vel_x_max=2.0 \
    task.initRandOpt.randDroneOptions.lin_vel_y_max=2.0 \
    task.initRandOpt.randDroneOptions.lin_vel_z_max=2.0 \
    task.initRandOpt.randDroneOptions.ang_vel_x_max=2.0 \
    task.initRandOpt.randDroneOptions.ang_vel_y_max=2.0 \
    task.initRandOpt.randDroneOptions.ang_vel_z_max=2.0 \
    task.initRandOpt.randDroneOptions.aileron_max=0.5 \
    task.initRandOpt.randDroneOptions.elevator_max=0.5 \
    task.initRandOpt.randDroneOptions.rudder_max=0.5 \
    task.initRandOpt.randDroneOptions.throttle_min=-1.0 \
    task.initRandOpt.randDroneOptions.throttle_max=0.0 \
    train.params.network.mlp.units="[256, 128, 128, 64]" \
    train.params.config.normalize_input=False \
    checkpoint=$1
}

ulimit -n 65535

cd ../../../../../

run_out=$(run $1 $2)

# extract SH_LOG_DIR
log_dir=$(echo "$run_out" | grep 'SH_IO_LOG_DIR:' | cut -d':' -f2- | xargs)
exp_dir=$(dirname $log_dir)

cd tasks/drone_racing/demos/

# process logs, currently it requires logging also images
python process_logs.py \
  --exp_dir $exp_dir \
  --pcd_update_itv 25

python rerun_exp.py \
  --exp_dir $exp_dir \
  --combine_envs \
  --only_traj_combine \
  --vel_max_cmap 20
