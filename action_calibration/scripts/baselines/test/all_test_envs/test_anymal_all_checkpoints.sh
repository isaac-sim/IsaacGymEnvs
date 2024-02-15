train_seed=$1
test_seed=100

for env_id in ContextualAnymalTestEasy1 ContextualAnymalTestEasy2 ContextualAnymalTestEasy3 \
    ContextualAnymalTestEasy4 ContextualAnymalTestEasy5 ContextualAnymalTestEasy6 \
    ContextualAnymalTestEasy7 ContextualAnymalTestEasy8 ContextualAnymalTestEasy9 \
    ContextualAnymalTestHard1 ContextualAnymalTestHard2 ContextualAnymalTestHard3 ContextualAnymalTestHard4
do
    for run_name in Anymal ContextualAnymalTrain $env_id
    do
        run_dir="runs/Anymal/seed_"$train_seed"/"$run_name
        for checkpoint_path in $run_dir"/checkpoints/"*.pth
        do
            python -m src.test \
                --checkpoint_path $checkpoint_path \
                --env_id $env_id \
                --seed $test_seed
        done
    done
done

######################################################

run_name=ContextualAnymalTrain_with_oracle_sys_params
run_dir="runs/Anymal/seed_"$train_seed"/"$run_name

for env_id in ContextualAnymalTestEasy1 ContextualAnymalTestEasy2 ContextualAnymalTestEasy3 \
    ContextualAnymalTestEasy4 ContextualAnymalTestEasy5 ContextualAnymalTestEasy6 \
    ContextualAnymalTestEasy7 ContextualAnymalTestEasy8 ContextualAnymalTestEasy9 \
    ContextualAnymalTestHard1 ContextualAnymalTestHard2 ContextualAnymalTestHard3 ContextualAnymalTestHard4
do
    for checkpoint_path in $run_dir"/checkpoints/"*.pth
    do
        python -m src.test_with_oracle_sys_params \
            --checkpoint_path $checkpoint_path \
            --env_id $env_id \
            --seed $test_seed
    done
done