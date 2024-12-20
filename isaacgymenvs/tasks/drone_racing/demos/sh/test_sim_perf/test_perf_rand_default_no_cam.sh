run() {
  python train.py task=DRRandom \
    headless=True \
    max_iterations=6 \
    num_envs=$1 \
    train.params.config.horizon_length=$2 \
    train.params.config.minibatch_size=$3 \
    task.env.obsImgMode=empty \
    task.env.enableCameraSensors=False \
    task.env.numObservations=56
}

cd ../../../../../

run 1 50 10
echo "finished: random default, 1, no cam, vram=880"

run 512 50 5120
echo "finished: random default, 512, no cam, vram=1272"

run 1024 50 10240
echo "finished: random default, 1024, no cam, vram=1732"

run 2048 50 20480
echo "finished: random default, 2048, no cam, vram=2368"

run 4096 50 40960
echo "finished: random default, 4096, no cam, vram=3822"

run 8192 50 81920
echo "finished: random default, 8192, no cam, vram=6710"

run 16384 50 163840
echo "finished: random default, 16384, no cam, vram=12356"

run 24576 50 245760
echo "finished: random default, 24576, no cam, vram=17554"
