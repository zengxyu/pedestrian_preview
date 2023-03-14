import os.path
import pickle

parent_path = "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/p2v/sg_walls/test/v_forces"
filename = "env_0.pkl"
path = os.path.join(parent_path, filename)
obj = pickle.load(open(path, "rb"))
out_parent_path = "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/p2v/sg_walls/test/v_image_forces"
out_path = os.path.join(out_parent_path, filename)

