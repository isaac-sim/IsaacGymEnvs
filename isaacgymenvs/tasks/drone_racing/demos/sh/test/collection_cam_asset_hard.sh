ts=$(date +"%Y%m%d%H%M%S")

./test_cam_multistory.sh $1 "${ts}_multistory"
./test_cam_rmua.sh $1 "${ts}_rmua"
./test_cam_walls.sh $1 "${ts}_walls"
./test_cam_wavy_eight.sh $1 "${ts}_wavy_eight"
