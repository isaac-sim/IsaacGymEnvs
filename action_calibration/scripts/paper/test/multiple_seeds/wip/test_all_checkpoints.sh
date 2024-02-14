run_dir=$1
env_id=$2
seed=$3

for checkpoint_path in $run_dir"/checkpoints/"*.pth
do
    python -m src.test \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $seed
done