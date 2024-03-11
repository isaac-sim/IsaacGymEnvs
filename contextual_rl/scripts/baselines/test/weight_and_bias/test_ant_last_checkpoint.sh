env_id=$1
train_seed=0
test_seed=100

for run_name in Ant ContextualAntTrain $env_id
do
    run_dir="runs/training/seed_"$train_seed"/"$run_name
    checkpoint_path=$run_dir"/checkpoints/99942400.pth"
    python -m src.test_ppo \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed \
        --track \
        --wandb_project_name Ant_test_$env_id \
        --wandb_entity cw-kang \
        --capture_video
done

######################################################

run_name=ContextualAntTrain_osi_true
run_dir="runs/training/seed_"$train_seed"/"$run_name
checkpoint_path=$run_dir"/checkpoints/99942400.pth"
python -m src.test_osi_true \
    --checkpoint_path $checkpoint_path \
    --env_id $env_id \
    --seed $test_seed \
    --track \
    --wandb_project_name Ant_test_$env_id \
    --wandb_entity cw-kang \
    --capture_video