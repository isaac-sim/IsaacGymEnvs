#!/bin/bash

train.py seed=-1 pbt=pbt_default pbt.workspace=workspace_allegro_kuka pbt.interval_steps=20000000 pbt.start_after=100000000 pbt.initial_delay=200000000 pbt.replace_fraction_worst=0.3 pbt/mutation=allegro_kuka_juggle_mutation task=AllegroKukaLSTM task/env=juggle pbt.num_policies=8 pbt.policy_idx=0 wandb_activate=True robojuggler


