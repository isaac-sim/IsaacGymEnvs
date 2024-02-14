checkpoint_path=$1
env_id=$2
seed=$3

python -m src.test \
    --checkpoint_path $checkpoint_path \
    --env_id $env_id \
    --seed $seed