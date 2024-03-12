train_seed=$1
test_seed=100

train_env_id=ContextualAntTrain

for env_id in ContextualAntTestEasy1 ContextualAntTestEasy2 ContextualAntTestEasy3 \
    ContextualAntTestEasy4 ContextualAntTestEasy5 ContextualAntTestEasy6 \
    ContextualAntTestEasy7 ContextualAntTestEasy8 ContextualAntTestEasy9 \
    ContextualAntTestHard1 ContextualAntTestHard2 ContextualAntTestHard3 ContextualAntTestHard4
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