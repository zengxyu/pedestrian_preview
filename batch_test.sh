gpu=0
in_folder="test_models/temporal_mass"
out_folder='test_result'
num_episodes=1000
model_index=3000

# receive shell parameters
show_usage="args: no thing"
GETOPT_ARGS=$(getopt -o g:i:o:n:mi -al gpu:,in_folder:,out_folder:,num_episodes:,model_index: -- "$@")
eval set -- "$GETOPT_ARGS"

while [ -n "$1" ]; do
  case "$1" in
  --gpu)
    gpu=$2
    shift 2
    ;;
  --in_folder)
    in_folder=$2
    shift 2
    ;;
  --out_folder)
    out_folder=$2
    shift 2
    ;;
  --num_episodes)
    num_episodes=$2
    shift 2
    ;;
  --model_index)
    model_index=$2
    shift 2
    ;;
  --) break ;;
  *)
    echo "$1","$2","$show_usage"
    break
    ;;
  esac
done

#office="office"
#intersection="intersection"
#corridor="corridor"
#hybrid="hybrid"
#
#if [ "$env" == "office" ]; then
#  env_probs="0 0 0 1"
#elif [ "$env" == "intersection" ]; then
#  env_probs="0 0 1 0"
#elif [ "$env" == "corridor" ]; then
#  env_probs="0 1 0 0"
#elif [ "$env" == "hybrid" ]; then
#  env_probs="0 1 1 1"
#else
#  echo "ERROR ================================="
#fi

static_nums=(0 1 2)
dynamic_nums=(1 2 3 4)
test_velocities=(0.1 0.2 0.4 0.5 0.6)
model_index=3000
env_probs="0 1 1 1"

# test static obstacle num
for static in ${static_nums[*]}; do
  echo "========= 测试静态人数： $static ================="
  python run.py --env pybullet --in_folder="$in_folder" --in_model_index="$model_index" --max_speed=0.3 --dynamic=0 --static_num="$static" --env_probs "$env_probs" --num_episodes="$num_episodes" --out_folder="$out_folder" --gpu="$gpu"
done

for dynamic in ${dynamic_nums[*]}; do
  echo "========= 测试动态人数： $dynamic ================="
  python run.py --env pybullet --in_folder="$in_folder" --in_model_index="$model_index" --max_speed=0.3 --dynamic="$dynamic" --static_num=1 --env_probs "$env_probs" --num_episodes="$num_episodes" --out_folder="$out_folder" --gpu="$gpu"
done

for speed in ${test_velocities[*]}; do
  echo "========= 测试速度： $speed ================="
  python run.py --env pybullet --in_folder="$in_folder" --in_model_index="$model_index" --max_speed="$speed" --dynamic=2 --static_num=1 --env_probs "$env_probs" --num_episodes="$num_episodes" --out_folder="$out_folder" --gpu="$gpu"
done

#env_probs_list=("0 0 0 1" "0 0 1 0" "0 1 0 0")
#for prob in ${env_probs_list[*]}; do
#  echo "========= 测试静态人数： $static ================="
#  python run.py --env pybullet --in_folder="$in_folder" --in_model_index="$model_index" --max_speed=0.3 --dynamic=2 --static_num=1 --env_probs "$prob" --num_episodes="$num_episodes" --out_folder="$out_folder"
#done
