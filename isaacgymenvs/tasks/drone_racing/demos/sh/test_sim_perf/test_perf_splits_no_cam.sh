run() {
  python train.py task=DRAsset \
    headless=True \
    max_iterations=6 \
    num_envs=$1 \
    train.params.config.horizon_length=$2 \
    train.params.config.minibatch_size=$3
}

cd ../../../../../

run 1 50 10
echo "finished: splits, 1, no cam, vram=880"

run 512 50 5120
echo "finished: splits, 512, no cam, vram=1204"

run 1024 50 10240
echo "finished: splits, 1024, no cam, vram=1528"

run 2048 50 20480
echo "finished: splits, 2048, no cam, vram=2168"

run 4096 50 40960
echo "finished: splits, 4096, no cam, vram=3432"

run 8192 50 81920
echo "finished: splits, 8192, no cam, vram=5940"

run 16384 50 163840
echo "finished: splits, 16384, no cam, vram=10960"

run 24576 50 245760
echo "finished: splits, 24576, no cam, vram=16104"
