# usage: ./plot_train_rand_dr.sh 252 280 6.5

python ../../plot_train_log.py \
  --ep_len_csv ../../train_log/rand_dr_ep_len.csv \
  --rew_csv ../../train_log/rand_dr_rew.csv \
  --rew_col_csv ../../train_log/rand_dr_rew_col.csv \
  --rew_wp_csv ../../train_log/rand_dr_rew_wp.csv \
  --fig_w $1 \
  --fig_h $2 \
  --fig_file ~/Desktop/train_log_rand_dr.pdf \
  --font_size $3 \
  --font_family serif \
  --xlim_high 530000000
