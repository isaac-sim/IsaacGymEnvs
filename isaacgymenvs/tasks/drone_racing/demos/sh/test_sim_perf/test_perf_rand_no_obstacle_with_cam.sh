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
    task.env.enableCameraSensors=True \
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

run 1 6 50 10 False False False False
echo "finished: random no obstacle, 1, cam, vram=1435"

run 200 6 50 2000 False False False False
echo "finished: random no obstacle, 200, cam, vram=4171"

run 400 6 50 4000 False False False False
echo "finished: random no obstacle, 400, cam, vram=6975"

run 600 6 50 6000 False False False False
echo "finished: random no obstacle, 600, cam, vram=9726"

run 800 6 50 8000 False False False False
echo "finished: random no obstacle, 800, cam, vram=12467"

run 1000 6 50 10000 False False False False
echo "finished: random no obstacle, 1000, cam, vram=15220"

run 1200 6 50 12000 False False False False
echo "finished: random no obstacle, 1200, cam, vram=17973"

run 1400 6 50 14000 False False False False
echo "finished: random no obstacle, 1400, cam, vram=20782"
