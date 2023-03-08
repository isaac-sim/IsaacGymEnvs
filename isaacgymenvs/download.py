""" Simple script to download WandB checkpoint"""

import pathlib
import argparse 
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-path", type=str)
    parser.add_argument("--wandb-filename", type=str)

    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(f"{args.wandb_run_path}")
    model = run.file(args.wandb_filename)
    model.download(".", replace=True)
    
    # Move the download to new directory
    download_fp = pathlib.Path(args.wandb_filename)
    output_dir = pathlib.Path('downloads')
    output_dir = output_dir / str(run.id)
    output_dir.mkdir(exist_ok=True, parents=True) 
    output_fp = output_dir / download_fp.name
    download_fp.rename(output_fp)