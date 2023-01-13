# nav-learning


## Install requirements
```
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip3 install -e .
```
## Compile
```angular2html
sh build.sh
```

### Run motion agent

#### 1. Train agent
Train the agent, specifying
```
python3 run.py --train  --env pybullet --out_folder=out_motion_folder
```

#### 2. Test agent
--render if you wanna visualize it
```
python3 run.py --render --env pybullet --in_folder=temporal_mass --in_model_index=3000 --num_episodes=1000
```

##### 1. Test agent
```

```

#### human walking and collision detection
```
https://github.com/epfl-lasa/human-robot-collider.git
```