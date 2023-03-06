# nav-learning


## Prerequisites
```
pip3 install -r requirements.txt
```

## Compile A* global planner
```commandline
python setup.py build_ext --inplace
```

#### 1. Train agent
Train the agent, specifying
```
python3 run.py --train --out_folder=[] --gpu=[]
```

#### 2. Resume training
```
python3 run.py --train --resume --in_folder=[] --in_model_index=[] --out_folder=[] --gpu=[]
```

#### 2. Test agent
--render if you wanna visualize it
```
python3 run.py --render  --in_folder=[] --in_model_index=[model_index] --num_episodes=[]
```

## Generate your office_1500 dataset

```commandline
python3 gen_fixed_envs.py
```

## compute geodesic distance by multi process
```commandline
compute_geodesic_distance.py
```

## compute min distance map
```commandline
python3 compute_min_obstacle_distance.py
```