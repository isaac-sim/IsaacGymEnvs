# data in this script is collected using
# test_perf_rand_no_obstacle_no_cam.sh and test_perf_rand_no_obstacle_with_cam.sh
# OS: Ubuntu 22.04.4 LTS x86_64
# CPU: 13th Gen Intel i9-13900K (32) @ 5.500GHz
# GPU: NVIDIA RTX 4090
# arg 1: path to environment screenshot
# arg 2: figure width in pt
# arg 3: figure height in pt
# arg 4: file of saved figure
# arg 5: font size
# arg 6: font family
# arg 7: additional y axis offset
# arg 8: marker size
# arg 9: result img dpi
# arg 10: legend vertical space
# usage:
# ./plot_perf_rand_no_obstacle.sh ../../imgs/perf_test_rand_no_obst.png 460 110 ~/Desktop/perf_rand_no_obst.pdf 6.3 sans-serif 1.15 4 1000 0.25

python ../../plot_perf_test.py \
  --env_img $1 \
  --num_envs_no_cam 1 512 1024 2048 4096 8192 16384 24576 \
  --total_fps_no_cam 111 42826 78176 138784 237917 351394 430363 442050 \
  --vram_no_cam 880 1204 1466 2106 3272 5498 9990 14382 \
  --num_envs_cam 1 200 400 600 800 1000 1200 1400 \
  --total_fps_cam 87 4129 4643 4755 4752 4706 4635 4548 \
  --vram_cam 1435 4171 6975 9726 12467 15220 17973 20782 \
  --fig_w $2 \
  --fig_h $3 \
  --fig_file $4 \
  --font_size $5 \
  --font_family $6 \
  --yaxis_offset $7 \
  --marker_size $8 \
  --dpi $9 \
  --legend_vspace ${10}
