run() {
  python train.py task=DRRandom \
    num_envs=$1 \
    max_iterations=$2 \
    train.params.config.horizon_length=$3 \
    train.params.config.minibatch_size=$4 \
    headless=$5 \
    wandb_activate=$6 \
    task.disableObstacleManager=$7 \
    task.env.enableDebugVis=$8 \
    task.env.numObservations=56 \
    task.env.obsImgMode=empty \
    task.env.enableCameraSensors=False \
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
    task.envCreator.num_wall_assets=0
}

cd ../../../../../

run 1 6 50 10 True False False False
echo "finished: random no obstacle, 1, no cam, vram=880"

run 512 6 50 5120 True False False False
echo "finished: random no obstacle, 512, no cam, vram=1204"

run 1024 6 50 10240 True False False False
echo "finished: random no obstacle, 1024, no cam, vram=1466"

run 2048 6 50 20480 True False False False
echo "finished: random no obstacle, 2048, no cam, vram=2106"

run 4096 6 50 40960 True False False False
echo "finished: random no obstacle, 4096, no cam, vram=3272"

run 8192 6 50 81920 True False False False
echo "finished: random no obstacle, 8192, no cam, vram=5498"

run 16384 6 50 163840 True False False False
echo "finished: random no obstacle, 16384, no cam, vram=9990"

run 24576 6 50 245760 True False False False
echo "finished: random no obstacle, 24576, no cam, vram=14382"
