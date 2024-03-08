for seed in 0 1 2 3 4
do
    env_id=Anymal
    run_dir="runs/training/seed_"$seed"/"$env_id
    checkpoint_path=$run_dir"/checkpoints/99942400.pth"
    python -m src.train_inverse_dynamics_model \
        --env_id $env_id \
        --seed $seed \
        --checkpoint_path $checkpoint_path \
        --total_timesteps 100000000 \
        --anneal_lr
done