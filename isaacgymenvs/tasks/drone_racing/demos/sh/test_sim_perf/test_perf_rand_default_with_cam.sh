run() {
  python train.py task=DRRandom \
    headless=False \
    max_iterations=6 \
    num_envs=$1 \
    train.params.config.horizon_length=$2 \
    train.params.config.minibatch_size=$3 \
    task.env.obsImgMode=empty \
    task.env.enableCameraSensors=True \
    task.env.numObservations=56
}

cd ../../../../../

run 1 50 10
echo "finished: random default, 1, cam, vram=1438"

run 200 50 2000
echo "finished: random default, 200, cam, vram=4246"

run 400 50 4000
echo "finished: random default, 400, cam, vram=7058"

run 600 50 6000
echo "finished: random default, 600, cam, vram=9888"

run 800 50 8000
echo "finished: random default, 800, cam, vram=12628"

run 1000 50 10000
echo "finished: random default, 1000, cam, vram=15513"

run 1200 50 12000
echo "finished: random default, 1200, cam, vram=18298"

run 1400 50 14000
echo "finished: random default, 1400, cam, vram=21043"
