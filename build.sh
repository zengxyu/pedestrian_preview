cp traditional_planner/a_star/astar.py traditional_planner/a_star/astar.pyx

cp environment/nav_utilities/bubble_utils.py environment/nav_utilities/bubble_utils.pyx
cp environment/nav_utilities/icp.py environment/nav_utilities/icp.pyx

rm -rf traditional_planner/*/*.c
rm -rf traditional_planner/*/*.so
rm -rf environment/nav_utilities/*.c
rm -rf environment/nav_utilities/*.so

python3 setup.py build_ext --inplace

rm -rf build
