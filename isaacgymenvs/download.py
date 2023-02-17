""" Simple script to download WandB checkpoint"""

import os
import argparse 
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-run-id", type=str)
    parser.add_argument("--wandb-filename", type=str)

    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}")
    model = run.file(args.wandb_filename)
    model.download(".")