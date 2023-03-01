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
