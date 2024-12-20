if [ "$#" -ne 1 ]; then
    echo "Error: incorrect arg count"
    echo "Require: checkpoint"
    echo "Example: ./test_no_cam_splits.sh CHECKPOINT_PATH"
    exit 1
fi

cd ../../../../../

ts=$(date +"%Y%m%d%H%M%S")
exp_name="${ts}_test_splits"

run_out=$(
  python train.py task=DRAsset \
    test=True \
    num_envs=1 \
    task.assetName=splits \
    task.env.appendWpDist=0 \
    task.env.disableGround=True \
    task.env.enableCameraSensors=True \
    task.env.maxEpisodeLength=1000 \
    task.env.logging.enable=True \
    task.env.logging.experimentName=$exp_name \
    task.env.logging.logMainCam=True \
    task.env.logging.logExtraCams=True \
    task.env.logging.maxNumEpisodes=1 \
    task.env.logging.numStepsPerSave=1000 \
    task.droneSim.drone_asset_options.disable_visuals=True \
    train.params.network.mlp.units="[256, 128, 128, 64]" \
    train.params.config.normalize_input=False \
    checkpoint=$1
)

ulimit -n 65535

# extract SH_LOG_DIR
log_dir=$(echo "$run_out" | grep 'SH_IO_LOG_DIR:' | cut -d':' -f2- | xargs)
exp_dir=$(dirname $log_dir)

cd tasks/drone_racing/demos/

# process logs, currently it requires logging also images
python process_logs.py \
  --exp_dir $exp_dir \
  --pcd_update_itv 10

# rerun experiment
python rerun_exp.py \
  --exp_dir $exp_dir \
  --combine_envs
