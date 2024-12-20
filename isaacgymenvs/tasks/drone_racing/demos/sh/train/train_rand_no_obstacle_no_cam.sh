# make sure in virtual env

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
  echo "Error: incorrect arg count"
  echo "Require: headless wandb max_iter [checkpoint]"
  echo "Example: ./train_rand_no_obstacle_no_cam.sh True True 150"
  exit 1
fi

cd ../../../../../

python train.py task=DRRandom \
  num_envs=16384 \
  headless=$1 \
  wandb_activate=$2 \
  max_iterations=$3 \
  task.disableObstacleManager=False \
  task.env.numObservations=56 \
  task.env.obsImgMode=empty \
  task.env.enableCameraSensors=False \
  task.env.enableDebugVis=False \
  task.envCreator.num_box_actors=0 \
  task.envCreator.num_box_assets=0 \
  task.envCreator.num_capsule_actors=0 \
  task.envCreator.num_capsule_assets=0 \
  task.envCreator.num_cuboid_wireframe_actors=0 \
  task.envCreator.num_cuboid_wireframe_assets=0 \
  task.envCreator.num_cylinder_actors=0 \
  task.envCreator.num_cylinder_assets=0 \
  task.envCreator.num_hollow_cuboid_actors=0 \
  task.envCreator.num_hollow_cuboid_assets=0 \
  task.envCreator.num_sphere_actors=0 \
  task.envCreator.num_sphere_assets=0 \
  task.envCreator.num_tree_actors=0 \
  task.envCreator.num_tree_assets=0 \
  task.envCreator.num_wall_actors=0 \
  task.envCreator.num_wall_assets=0 \
  task.initRandOpt.randWaypointOptions.init_roll_max=0.5 \
  task.initRandOpt.randWaypointOptions.init_pitch_max=0.5 \
  task.initRandOpt.randWaypointOptions.init_yaw_max=3.14 \
  task.initRandOpt.randWaypointOptions.psi_max=1.57 \
  task.initRandOpt.randWaypointOptions.theta_max=1.57 \
  task.initRandOpt.randWaypointOptions.alpha_max=3.14 \
  task.initRandOpt.randWaypointOptions.gamma_max=0.5 \
  task.initRandOpt.randWaypointOptions.r_min=2.0 \
  task.initRandOpt.randWaypointOptions.r_max=18.0 \
  train.params.network.mlp.units="[256, 128, 128, 64]" \
  train.params.config.normalize_input=False \
  train.params.config.horizon_length=100 \
  train.params.config.minibatch_size=102400 \
  checkpoint=$4
