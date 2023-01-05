dynamic_num=2
max_speed=0.35
num_episodes=1000

python3 run.py --env_probs 0 0 0 1 --dynamic_num=${dynamic_num} --static_num=0  --max_speed=${max_speed} --num_episodes=${num_episodes} --in_folder=expand_office_test_model/speed_control_adjust --in_model_index=3600 --out_folder=evaluation_speed_control_adjust &
python3 run.py --env_probs 0 0 1 0 --dynamic_num=${dynamic_num} --static_num=0  --max_speed=${max_speed} --num_episodes=${num_episodes} --in_folder=expand_office_test_model/speed_control_adjust --in_model_index=3600 --out_folder=evaluation_speed_control_adjust &
python3 run.py --env_probs 0 1 0 0 --dynamic_num=${dynamic_num} --static_num=0  --max_speed=${max_speed} --num_episodes=${num_episodes} --in_folder=expand_office_test_model/speed_control_adjust --in_model_index=3600 --out_folder=evaluation_speed_control_adjust &
python3 run.py --env_probs 0 1 1 1 --dynamic_num=${dynamic_num} --static_num=0  --max_speed=${max_speed} --num_episodes=${num_episodes} --in_folder=expand_office_test_model/speed_control_adjust --in_model_index=3600 --out_folder=evaluation_speed_control_adjust &
python3 run.py --env real --in_folder=temporal_mass --in_model_index=3000 --num_episodes=1000
python3 run.py --env pybullet  --in_folder=temporal_mass --in_model_index=3000  --max_speed 0.4 --env_probs="0 0 0 1"  --num_episodes=500  --render