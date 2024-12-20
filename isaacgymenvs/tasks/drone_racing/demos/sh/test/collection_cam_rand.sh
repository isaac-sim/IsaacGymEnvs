ts=$(date +"%Y%m%d%H%M%S")

#./test_cam_rand_l0.sh $1 "${ts}_rand_l0"  # too similar to l1
./test_cam_rand_l1.sh $1 "${ts}_rand_l1"
#./test_cam_rand_l2.sh $1 "${ts}_rand_l2"  # too similar to l1
./test_cam_rand_l3.sh $1 "${ts}_rand_l3"
./test_cam_rand_l4.sh $1 "${ts}_rand_l4"
./test_cam_rand_l5.sh $1 "${ts}_rand_l5"
