run_dir=$1
env_id=$2
seed=$3

for checkpoint_path in $run_dir"/checkpoints/"*.pth
do
    python -m src.test \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $seed \
        --track \
        --wandb_project_name action_calibration \
        --wandb_entity cw-kang \
        --capture_video
done