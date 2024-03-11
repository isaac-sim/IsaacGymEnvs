for seed in 0 1 2 3 4
do
    for env_id in Anymal \
        ContextualAnymalTrain \
        ContextualAnymalTestEasy1 ContextualAnymalTestEasy2 ContextualAnymalTestEasy3 \
        ContextualAnymalTestEasy4 ContextualAnymalTestEasy5 ContextualAnymalTestEasy6 \
        ContextualAnymalTestEasy7 ContextualAnymalTestEasy8 ContextualAnymalTestEasy9 \
        ContextualAnymalTestHard1 ContextualAnymalTestHard2 ContextualAnymalTestHard3 ContextualAnymalTestHard4
    do
        python -m src.train_ppo \
            --env_id $env_id \
            --seed $seed \
            --total_timesteps 100000000 \
            --anneal_lr
    done
done