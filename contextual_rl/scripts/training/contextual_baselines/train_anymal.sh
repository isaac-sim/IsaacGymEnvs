for seed in 0 1 2 3 4
do
    env_id=ContextualAnymalTrain

    python -m src.train_ppo_stacked \
        --env_id $env_id \
        --seed $seed \
        --total_timesteps 100000000 \
        --anneal_lr

    python -m src.train_osi_true \
        --env_id $env_id \
        --seed $seed \
        --total_timesteps 100000000 \
        --anneal_lr

    for baseline_phase1 in osi_phase1 dm_phase1
    do
        python -m "src.train_"$baseline_phase1 \
            --env_id $env_id \
            --seed $seed \
            --total_timesteps 100000000 \
            --anneal_lr \
            --checkpoint_path "runs/training/seed_"$seed"/"$env_id"_osi_true/checkpoints/99942400.pth"
    done

    python -m src.train_osi_phase2 \
        --env_id $env_id \
        --seed $seed \
        --total_timesteps 100000000 \
        --anneal_lr \
        --checkpoint_path "runs/training/seed_"$seed"/"$env_id"_osi/checkpoints/99942400_phase1.pth"
    
    python -m src.train_osi_phase2_raw \
        --env_id $env_id \
        --seed $seed \
        --total_timesteps 100000000 \
        --anneal_lr \
        --checkpoint_path "runs/training/seed_"$seed"/"$env_id"_osi/checkpoints/99942400_phase1.pth"

    python -m src.train_dm_phase2 \
        --env_id $env_id \
        --seed $seed \
        --total_timesteps 100000000 \
        --anneal_lr \
        --checkpoint_path "runs/training/seed_"$seed"/"$env_id"_dm/checkpoints/99942400_phase1.pth"
done