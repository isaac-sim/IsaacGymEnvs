Videos available at: https://drive.google.com/drive/folders/1-Zt699xr16caMgP0ZjBXfwnJK1eVLZq2?usp=share_link

Training `getup.pth` (`getup_from_random_pose.mp4`, `getup_from_back.mp4`): 

```
python train.py task=QuadrupedAMP headless=True max_iterations=5000 task.env.enableEarlyTermination=False task.env.episodeLength_s=2.5 task.env.enableRefStateInitHeight=True task.env.motionFile=data/motions/quadruped/a1_expert/dataset_recover.yaml task.env.stateInit=Hybrid wandb_entity=dtch1997 wandb_project=QuadrupedASE wandb_group=QuadrupedAMPGetup wandb_activate=True wandb_name=all_recover 
```

Training `getup_locomotion.pth` (`getup_locomotion.mp4`): 

```
python train.py task=QuadrupedAMP task.env.motionFile=data/motions/quadruped/a1_expert/dataset_locomotion_recovery.yaml headless=True wandb_entity=dtch1997 wandb_project=QuadrupedASE wandb_group=QuadrupedGaits wandb_activate=True max_iterations=4000 task.env.enableEarlyTermination=False task.env.episodeLength_s=4 task.env.enableRefStateInitHeight=True task.env.stateInit=Hybrid wandb_name=recovery_locomotion
```

