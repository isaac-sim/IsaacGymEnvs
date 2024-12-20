ts=$(date +"%Y%m%d%H%M%S")

./test_cam_simple_stick_no_obst.sh $1 "${ts}_simple_stick_no_obst"
./test_cam_geom_kebab_no_obst.sh $1 "${ts}_geom_kebab_no_obst"
./test_cam_planar_circle_no_obst.sh $1 "${ts}_planar_circle_no_obst"
./test_cam_wavy_eight_no_obst.sh $1 "${ts}_wavy_eight_no_obst"
./test_cam_turns.sh $1 "${ts}_turns"
