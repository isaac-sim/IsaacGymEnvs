env_id=$1
train_seed=0
test_seed=100

for run_name in Anymal ContextualAnymalTrain $env_id
do
    run_dir="runs/Anymal/seed_"$train_seed"/"$run_name
    checkpoint_path=$run_dir"/checkpoints/99942400.pth"
    python -m src.test \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed \
        --track \
        --wandb_project_name Anymal_test_$env_id \
        --wandb_entity cw-kang \
        --capture_video
done

######################################################

run_name=ContextualAnymalTrain_with_oracle_sys_params
run_dir="runs/Anymal/seed_"$train_seed"/"$run_name
checkpoint_path=$run_dir"/checkpoints/99942400.pth"
python -m src.test_with_oracle_sys_params \
    --checkpoint_path $checkpoint_path \
    --env_id $env_id \
    --seed $test_seed \
    --track \
    --wandb_project_name Anymal_test_$env_id \
    --wandb_entity cw-kang \
    --capture_video