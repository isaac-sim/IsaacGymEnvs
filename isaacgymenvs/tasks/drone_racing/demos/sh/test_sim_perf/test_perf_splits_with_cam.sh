run() {
  python train.py task=DRAsset \
    headless=False \
    max_iterations=6 \
    num_envs=$1 \
    train.params.config.horizon_length=$2 \
    train.params.config.minibatch_size=$3 \
    task.env.enableCameraSensors=True
}

cd ../../../../../

run 1 50 10
echo "finished: splits, 1, cam, vram=1435"

run 200 50 2000
echo "finished: splits, 200, cam, vram=4238"

run 400 50 4000
echo "finished: splits, 400, cam, vram=6982"

run 600 50 6000
echo "finished: splits, 600, cam, vram=9799"

run 800 50 8000
echo "finished: splits, 800, cam, vram=12543"

run 1000 50 10000
echo "finished: splits, 1000, cam, vram=15302"

run 1200 50 12000
echo "finished: splits, 1200, cam, vram=18182"

run 1400 50 14000
echo "finished: splits, 1400, cam, vram=20926"
