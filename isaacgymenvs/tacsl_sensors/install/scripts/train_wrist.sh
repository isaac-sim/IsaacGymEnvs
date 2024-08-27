#!/bin/bash

cd $( dirname ${BASH_SOURCE[0]} )/../../../


python train.py \
    'task=TacSLTaskInsertion' 'task.env.task_type=insertion' \
    'seed=-1' 'task.env.use_gelsight=True' 'headless=True' \
    'max_iterations=1500' 'task.env.numEnvs=64' \
    'task.randomize.plug_noise_rot_in_gripper=[0.0, 0.628318, 0.0]' \
    'train=TacSLTaskInsertionPPO_LSTM_dict_AAC' 'train.params.config.horizon_length=512' \
    'train.params.config.mini_epochs=4' \
    +'task.env.obsDims={ee_pos:[3],ee_quat:[4],socket_pos:[3],socket_quat:[4],dof_pos:[9]}' \
    +'train.params.network.input_preprocessors={ee_pos:{},ee_quat:{},socket_pos:{},socket_quat:{},dof_pos:{}}' \
    'task.rl.asymmetric_observations=True' 'task.rl.add_contact_info_to_aac_states=True' \
    +'task.env.stateDims={ee_pos:[3],ee_quat:[4],plug_pos:[3],plug_quat:[4],socket_pos_gt:[3],socket_quat:[4],dof_pos:[9],ee_lin_vel:[3],ee_ang_vel:[3],plug_socket_force:[3],plug_left_elastomer_force:[3],plug_right_elastomer_force:[3]}' \
    +'train.params.config.central_value_config.network.input_preprocessors={ee_pos:{},ee_quat:{},plug_pos:{},plug_quat:{},socket_pos_gt:{},socket_quat:{},dof_pos:{},ee_lin_vel:{},ee_ang_vel:{},plug_socket_force:{},plug_left_elastomer_force:{},plug_right_elastomer_force:{}}' \
    'task.rl.add_contact_force_plug_decomposed=True' \
    'task.env.use_camera_obs=True' 'task.env.use_camera=True' 'task.env.use_isaac_gym_tactile=False' \
    +'task.env.obsDims={wrist:[64,64,3]}' \
    +'train.params.network.input_preprocessors={wrist:{cnn:{type:conv2d_spatial_softargmax,activation:relu,initializer:{name:default},regularizer:{name:'None'},convs:[{filters:32,kernel_size:8,strides:2,padding:0},{filters:64,kernel_size:4,strides:1,padding:0},{filters:64,kernel_size:3,strides:1,padding:0}]}}}' \
    experiment=insert_wrist