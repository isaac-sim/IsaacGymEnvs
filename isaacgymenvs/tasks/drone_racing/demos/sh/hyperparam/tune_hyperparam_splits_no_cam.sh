run() {
  python train.py task=DRAsset \
    headless=True \
    wandb_activate=True \
    max_iterations=$1 \
    num_envs=$2 \
    train.params.network.mlp.units=$3 \
    train.params.config.normalize_input=$4 \
    train.params.config.horizon_length=$5 \
    train.params.config.minibatch_size=$6 \
    train.params.config.name=$7
}

cd ../../../../../

n_envs=16384
b_iters=400
for units in \
  "[256,256,256,256]" \
  "[256,128,128,64]" \
  "[128,128,128,128]" \
  "[256,256,256]" \
  "[256,128,64]" \
  "[128,128,128]" \
  "[256,256]" \
  "[256,128]" \
  "[128,128]"; do
  for normalize in False True; do
    for horizon in 50 100 200; do
      for denom in 8 16 32; do
        minibatch_size=$((n_envs * horizon / denom))
        clean_units=$(echo $units | tr -d "[]" | tr "," "-" | tr -d " ")
        name="${clean_units}_${normalize}_${horizon}_${denom}"
        iters=$((b_iters * 50 / horizon))
        echo $iters $name
        run $iters $n_envs $units $normalize $horizon $minibatch_size $name
      done
    done
  done
done
