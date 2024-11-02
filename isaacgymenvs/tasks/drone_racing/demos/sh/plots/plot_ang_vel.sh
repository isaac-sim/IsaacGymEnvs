# usage: ./plot_ang_vel.sh 460 350 ~/Desktop/ang_vel.pdf 8 sans-serif 0.05

cd ../../../../../

ts=$(date +"%Y%m%d%H%M%S")
exp_name="plot_ang_vel_$ts"

# run rollout on SPLIT-S with all cameras enabled
run_out=$(
  python train.py task=DRAsset \
    checkpoint=tasks/drone_racing/demos/checkpoints/splits_ang_vel_plot.pth \
    test=True \
    num_envs=1 \
    train.params.config.horizon_length=500 \
    task.env.enableCameraSensors=True \
    task.droneSim.drone_asset_options.disable_visuals=True \
    task.env.logging.enable=True \
    task.env.logging.experimentName=$exp_name \
    task.env.logging.logMainCam=True \
    task.env.logging.logExtraCams=True \
    task.env.logging.maxNumEpisodes=1 \
    task.env.logging.numStepsPerSave=500 \
    task.assetName=splits \
    task.env.disableGround=True
)

# extract SH_LOG_DIR
log_dir=$(echo "$run_out" | grep 'SH_IO_LOG_DIR:' | cut -d':' -f2- | xargs)
exp_dir=$(dirname $log_dir)

cd tasks/drone_racing/demos/

# process logs, currently it requires logging also images
python process_logs.py \
  --exp_dir $exp_dir

# plot
python plot_ang_vel.py \
  --log_file "$exp_dir/log_0.pt" \
  --episode_id 0 \
  --ctrl_dt 0.004 \
  --fig_w $1 \
  --fig_h $2 \
  --fig_file $3 \
  --font_size $4 \
  --font_family $5 \
  --legend_vspace $6
