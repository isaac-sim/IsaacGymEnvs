for seed in 0 1 2 3 4
do
    env_id=ContextualAntTrain

    python -m src.train_osi_true_student_phase1 \
        --env_id $env_id \
        --seed $seed \
        --total_timesteps 100000000 \
        --anneal_lr \
        --checkpoint_path "runs/training/seed_"$seed"/"$env_id"_osi_true/checkpoints/99942400.pth"

    python -m src.train_osi_true_student_phase2 \
        --env_id $env_id \
        --seed $seed \
        --total_timesteps 100000000 \
        --anneal_lr \
        --checkpoint_path "runs/training/seed_"$seed"/"$env_id"_osi_true_student/checkpoints/99942400_phase1.pth"
done