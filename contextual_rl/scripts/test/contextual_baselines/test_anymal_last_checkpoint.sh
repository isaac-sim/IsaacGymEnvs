train_seed=$1
test_seed=100

train_env_id=ContextualAnymalTrain

for env_id in ContextualAnymalTestEasy1 ContextualAnymalTestEasy2 ContextualAnymalTestEasy3 \
    ContextualAnymalTestEasy4 ContextualAnymalTestEasy5 ContextualAnymalTestEasy6 \
    ContextualAnymalTestEasy7 ContextualAnymalTestEasy8 ContextualAnymalTestEasy9 \
    ContextualAnymalTestHard1 ContextualAnymalTestHard2 ContextualAnymalTestHard3 ContextualAnymalTestHard4
do
    run_name=$train_env_id"_stacked"
    run_dir="runs/training/seed_"$train_seed"/"$run_name

    checkpoint_path=$run_dir"/checkpoints/99942400.pth"

    python -m src.test_ppo_stacked \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed

    run_name=$train_env_id"_osi_true"
    run_dir="runs/training/seed_"$train_seed"/"$run_name

    checkpoint_path=$run_dir"/checkpoints/99942400.pth"

    python -m src.test_osi_true \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed


    run_name=$train_env_id"_osi"
    run_dir="runs/training/seed_"$train_seed"/"$run_name

    osi_checkpoint_path=$run_dir"/checkpoints/99942400_phase1.pth"
    checkpoint_path=$run_dir"/checkpoints/99942400_phase2.pth"
    
    python -m src.test_osi \
        --osi_checkpoint $osi_checkpoint_path \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed

    checkpoint_path=$run_dir"/checkpoints/99942400_phase2_raw.pth"
    
    python -m src.test_osi_raw \
        --osi_checkpoint $osi_checkpoint_path \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed


    run_name=$train_env_id"_dm"
    run_dir="runs/training/seed_"$train_seed"/"$run_name

    dm_checkpoint_path=$run_dir"/checkpoints/99942400_phase1.pth"
    checkpoint_path=$run_dir"/checkpoints/99942400_phase2.pth"
    
    python -m src.test_dm \
        --dm_checkpoint $dm_checkpoint_path \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed
done