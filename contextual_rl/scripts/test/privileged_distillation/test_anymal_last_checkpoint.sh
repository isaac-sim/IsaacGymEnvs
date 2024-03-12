train_seed=$1
test_seed=100

train_env_id=ContextualAnymalTrain

for env_id in ContextualAntTestEasy1 ContextualAntTestEasy2 ContextualAntTestEasy3 \
    ContextualAntTestEasy4 ContextualAntTestEasy5 ContextualAntTestEasy6 \
    ContextualAntTestEasy7 ContextualAntTestEasy8 ContextualAntTestEasy9 \
    ContextualAntTestHard1 ContextualAntTestHard2 ContextualAntTestHard3 ContextualAntTestHard4
do
    run_name=$train_env_id"_osi_true_student"
    run_dir="runs/training/seed_"$train_seed"/"$run_name

    student_checkpoint_path=$run_dir"/checkpoints/99942400_phase1.pth"
    checkpoint_path=$run_dir"/checkpoints/99942400_phase2.pth"
    
    python -m src.test_osi_true_student \
        --student_checkpoint $student_checkpoint_path \
        --checkpoint_path $checkpoint_path \
        --env_id $env_id \
        --seed $test_seed
done