env_id=$1
train_seed=0
test_seed=100

for run_name in Anymal ContextualAnymalTrain $env_id
do
    run_dir="runs/training/seed_"$train_seed"/"$run_name
    for checkpoint_path in $run_dir"/checkpoints/"*.pth
    do
        python -m src.test_ppo \
            --checkpoint_path $checkpoint_path \
            --env_id $env_id \
            --seed $test_seed \
            --track \
            --wandb_project_name Anymal_test_all_checkpoints_$env_id \
            --wandb_entity cw-kang \
            --capture_video
    done
done

######################################################

run_name=ContextualAnymalTrain_osi_true
run_dir="runs/training/seed_"$train_seed"/"$run_name
for checkpoint_path in $run_dir"/checkpoints/"*.pth
do
    python -m src.test_osi_true \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed \
        --track \
        --wandb_project_name Anymal_test_all_checkpoints_$env_id \
        --wandb_entity cw-kang \
        --capture_video
done