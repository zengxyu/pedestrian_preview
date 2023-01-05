"""
Confugration of the arg-parser settings.
Control program flow / save path / iGibson scene / ...
"""
import os
import argparse
import random
import time

import h5py
import yaml

from igibson import configs_path


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp():
    return str(time.time())


def add_dict_to_h5py(f, dic):
    for key in dic.keys():
        if dic[key]:
            if isinstance(dic[key], dict):
                d = f.create_group(key)
                add_dict_to_h5py(d, dic[key])
            else:
                # f.attrs[key] = dic[key]
                f.create_dataset(key, data=dic[key])


def add_h5py_meta_data(h5py_file, config):
    with h5py.File(h5py_file, 'a') as f:
        metadata = f.create_group("metadata")

        metadata_config = metadata.create_group("config")
        metadata_yaml = metadata.create_group("yaml")

        # Add the program config data
        for key in config._default_values.keys():
            # metadata_config.attrs[key] = config[key]
            metadata_config.create_dataset(key, data=config[key])

        # Add iGibson configuration data
        add_dict_to_h5py(metadata_yaml, config.yaml_data)


class IgibsonConfig(dict):
    """
    Program Configuration Class.
    """
    _optional_actors = ["random", "astar", "brrt", "rl_policy"]

    # Default Value Only.  New items added MUST have a default value
    _default_values = {
        "gui": True,
        "save_path": "./output/data/",
        "runs": 3,
        "actor": "rl_policy",
        "seed": 42,
        "yaml": "configs/igibson_config/22_08_24__single_dynamic_pedestrian_nav.yaml",
        # "yaml": os.path.join(configs_path, "turtlebot_static_nav.yaml")
        "action_timestep": 1 / 5.,
        "physics_timestep": 1 / 120.,
        "pb": False,
        "rl_yaml": "__experiment_configs__/22_08_25__final.yml",
        "model_path": "",
    }

    # Help for the argparser
    _help = {
        "gui": "Set flag to display GUI",
        "runs": "Number of episodes to run",
        "save_path": "Where the trajectories from each episode are saved",
        "actor": f"Actor to use.  Current options are:   {_optional_actors}",
        "seed": "Set the seed of randomness",
        "yaml": "iGibson yaml config file",
        "action_timestep": "iGibson time between calling 'step'.  Must be a multiple of physics_timestep",
        "physics_timestep": "iGibson time between the physics simulation.  Must be a multiple of action_timestep",
        "pb": "Launch Pybullet GUI",
        "rl_yaml": "yaml file specific for rl training and model configs",
        "model_path": "The model path for the rl_policy actor to load from"
    }

    def __init__(self):
        """
        Create an Config object with default values
        Info on different initializers:
            https://stackoverflow.com/questions/5738470/whats-an-example-use-case-for-a-python-classmethod
        """
        for key in self._default_values.keys():
            self[key] = self._default_values[key]

        # Config must be initialized by calling:
        #   self.setup()
        self.initialized = False

    @classmethod
    def get_config(cls):
        """
        Create a config object from the command line.
        """
        instance = cls()
        # Run setup on the instance
        instance.setup()

        return instance

    def setup(self):
        """
        Setup things like creating the save directory.
        Allow access to elements in the config.
        """
        self.initialized = True

        # Seed random generator
        if self["seed"] is not None:
            set_random_seed(self["seed"])

        # Create the hdf5 save file
        if len(self["save_path"]) > 0:
            self["save_path"] = os.path.expanduser(self["save_path"])
            make_folder(self["save_path"])
            # Create the save file
            filename = "dataset_" + get_timestamp() + ".hdf5"
            self["save_file"] = os.path.join(self["save_path"], filename)
            # self["save_file"] = h5py.File(os.path.join(self["save_path"], filename), 'a')
        else:
            self["save_path"] = None
            self["save_file"] = None

        # Load the iGibson yaml file.  Store in self.yaml_data
        self["yaml"] = os.path.expanduser(self["yaml"])
        self.yaml_data = yaml.load(open(self["yaml"], "r"), Loader=yaml.FullLoader)

        if "action_freq" in self.yaml_data:
            self["action_timestep"] = self.yaml_data["action_timestep"]
            print("Config: action_timestep updated to {} s!".format(self["action_timestep"]))

        if "action_freq_robot" in self.yaml_data:
            self["action_timestep_robot"] = self.yaml_data["action_timestep"]
            print("Config: action_timestep_robot updated to {} s!".format(self["action_timestep_robot"]))

        if "physics_timestep" in self.yaml_data:
            self["physics_timestep"] = self.yaml_data["physics_timestep"]
            print("Config: physics_timestep updated to {} s!".format(self["physics_timestep"]))

        add_h5py_meta_data(self["save_file"], self)

    def __setitem__(self, key, item):
        """
        Ensure that the item has a key in the dictionary

        Interesting list of functions part of this class (since it inherits from dict):
            https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
        """
        # if key not in self._default_values.keys():
        #     raise KeyError("No Default Value for Config option:  ", key)

        # Note: self[key] = item, or infinite recursion
        super().__setitem__(key, item)

    def __getitem__(self, key):
        """
        Access items from the config dictionary
        """
        if not self.initialized:
            raise RuntimeError("Configuration Not Initialized.  Run Config.setup()")
        return super().__getitem__(key)


if __name__ == "__main__":
    # print("Loading config from command line")
    c = Config.get_config()
    print("Config Settings:  ")
    print(c)
