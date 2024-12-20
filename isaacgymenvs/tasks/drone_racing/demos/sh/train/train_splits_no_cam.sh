# make sure in virtual env

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    echo "Error: incorrect arg count"
    echo "Require: headless wandb max_iter episode_len [checkpoint]"
    echo "Example: ./train_splits_no_cam.sh True True 300 175"
    exit 1
fi

cd ../../../../../

python train.py task=DRAsset \
  headless=$1 \
  wandb_activate=$2 \
  max_iterations=$3 \
  num_envs=16384 \
  task.assetName=splits \
  task.env.maxEpisodeLength=$4 \
  task.env.appendWpDist=0 \
  task.mdp.reward.k_guidance=0 \
  train.params.network.mlp.units="[256, 128, 128, 64]" \
  train.params.config.normalize_input=False \
  train.params.config.horizon_length=50 \
  train.params.config.minibatch_size=51200 \
  checkpoint=$5
