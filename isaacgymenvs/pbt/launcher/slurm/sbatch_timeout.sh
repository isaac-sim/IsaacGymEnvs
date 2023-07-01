#!/bin/bash
conda activate your_env
cd ~/your_project_dir || exit

timeout $TIMEOUT $CMD
if [[ $$? -eq 124 ]]; then
    sbatch $PARTITION--gres=gpu:$GPU -c $CPU --parsable --output $FILENAME-slurm-%j.out $FILENAME
fi
