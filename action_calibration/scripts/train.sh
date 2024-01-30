env_id=$1
seed=$2

python -m src.cleanrl_ppo_train \
    --env_id $env_id \
    --seed $seed