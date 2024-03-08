seed=0

for env_id in Anymal \
    ContextualAnymalTrain \
    ContextualAnymalTestEasy1 ContextualAnymalTestEasy2 ContextualAnymalTestEasy3 \
    ContextualAnymalTestEasy4 ContextualAnymalTestEasy5 ContextualAnymalTestEasy6 \
    ContextualAnymalTestEasy7 ContextualAnymalTestEasy8 ContextualAnymalTestEasy9 \
    ContextualAnymalTestHard1 ContextualAnymalTestHard2 ContextualAnymalTestHard3 ContextualAnymalTestHard4
do
    python -m src.train \
        --env_id $env_id \
        --seed $seed \
        --total_timesteps 100000000 \
        --anneal_lr \
        --track \
        --wandb_project_name Anymal_training \
        --wandb_entity cw-kang \
        --capture_video
done

######################################################

env_id=ContextualAnymalTrain

python -m src.train_with_oracle_sys_params \
    --env_id $env_id \
    --seed $seed \
    --total_timesteps 100000000 \
    --anneal_lr \
    --track \
    --wandb_project_name Anymal_training \
    --wandb_entity cw-kang \
    --capture_video