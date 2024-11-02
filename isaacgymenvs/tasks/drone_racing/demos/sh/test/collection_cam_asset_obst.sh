ts=$(date +"%Y%m%d%H%M%S")

./test_cam_simple_stick.sh $1 "${ts}_simple_stick"
./test_cam_geom_kebab.sh $1 "${ts}_geom_kebab"
./test_cam_planar_circle.sh $1 "${ts}_planar_circle"
./test_cam_wavy_eight.sh $1 "${ts}_wavy_eight"
