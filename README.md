# nav-learning
Safe and smooth navigation is required for the
mobile robot around moving pedestrians through cluttered
indoor environment to the destination, when performing tasks
such as material handling, house chores, etc. In this paper,
we present a spatio-temporal attention and deep reinforcement
learning (DRL) based approach for the navigation in an
indoor environment with rapidly moving obstacles and limited
viewing using only modest planar 2D LIDAR on a mobile
TurtleBot, while not affecting the achievement of the goal. We
decompose the task into two subtasks: (1). Global navigation.
(2). Local obstacle avoidance and path following. The global
navigation subtask is resolved by traditional A* planner. For
obstacle avoidance and path following, we process the obtained
modest 2D range readings into a temporal representation
called temporal accumulation group descriptors, and a spa-
cial representation, which are taken as the input to a deep
deterministic policy gradient (DDPG) with the spatio-temporal
attention network. Then it controls the robot with the outputted
the continuous linear velocity and angular velocity. We have
integrated our algorithm with differential TurtleBot in a real-
world environments and evaluated the performance on multiple
metrics in various scenarios such as corridor, intersection and
office, etc. with differently dense moving obstacles. It is able
to avoid collisions swiftly and achieve the preset navigation
destination. It shows an excellent performance in comparison
to a state-of-the-art method and is transferable to previously
unseen environments.

## Prerequisites
```
pip3 install tensorflow==2.4.0 tensorflow_probability==0.12.1 tqdm gtimer ray ray[tune] dm-tree opencv-python
```

## Install Pybullet-Gym
```
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip3 install -e .
```
## Compile
```angular2html
sh build.sh
```

## Install agents_hrl
```
cd ~/path/to/agents_hrl   
pip3 install -e .
```
```
cd ~/path/to/gym_hrl   
pip3 install -e .
```
## Install RVO2 (ORCA implementation)
```
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
python3 setup.py build
python3 setup.py install
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