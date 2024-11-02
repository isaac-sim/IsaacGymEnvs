# usage: ./plot_train_splits_direct.sh 460 100 8

python ../../plot_train_log.py \
  --ep_len_csv ../../train_log/splits_direct_ep_len.csv \
  --rew_csv ../../train_log/splits_direct_rew.csv \
  --fig_w $1 \
  --fig_h $2 \
  --fig_file ~/Desktop/train_log_splits_direct.pdf \
  --font_size $3 \
  --font_family sans-serif \
  --xlim_high 250000000
