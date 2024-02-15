env_id=$1
train_seed=0
test_seed=100

for run_name in Ant ContextualAntTrain $env_id
do
    run_dir="runs/Ant/seed_"$train_seed"/"$run_name
    for checkpoint_path in $run_dir"/checkpoints/"*.pth
    do
        python -m src.test \
            --checkpoint_path $checkpoint_path \
            --env_id $env_id \
            --seed $test_seed \
            --track \
            --wandb_project_name ant_baselines_$env_id \
            --wandb_entity cw-kang \
            --capture_video
    done
done

######################################################

run_name=ContextualAntTrain_with_oracle_sys_params
run_dir="runs/Ant/seed_"$train_seed"/"$run_name
for checkpoint_path in $run_dir"/checkpoints/"*.pth
do
    python -m src.test_with_oracle_sys_params \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed \
        --track \
        --wandb_project_name ant_baselines_$env_id \
        --wandb_entity cw-kang \
        --capture_video
done