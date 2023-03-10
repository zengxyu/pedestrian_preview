import os

from utils.fo_utility import get_project_path


def test_data_folder_complete():
    parent_folder = "data"
    spaces = ["\n\t-", "\n\t\t-", "\n\t\t\t-", "-\n\t\t\t\t", "\n\t\t\t\t\t-"]
    office1000_parent_folder = os.path.join(get_project_path(), "data/office_evacuation")

    sub1_folders = ["office_evacuation"]
    sub2_folders = ["sg_walls", "sg_no_walls", "goal_at_door"]
    sub3_folders = ["train", "test"]
    sub4_folders = ["envs", "geodesic_distance", "obstacle_distance", "u_forces", "v_forces"]
    folder_structure = parent_folder
    for sub1_folder in sub1_folders:
        space1 = spaces[0]
        folder_structure += space1
        folder_structure += sub1_folder
        for sub2_folder in sub2_folders:
            space2 = spaces[1]
            folder_structure += space2
            folder_structure += sub2_folder
            for sub3_folder in sub3_folders:
                space3 = spaces[2]
                folder_structure += space3
                folder_structure += sub3_folder
                for sub4_folder in sub4_folders:
                    space4 = spaces[3]
                    folder_structure += space4
                    folder_structure += sub4_folder

    warning2 = "Your folder structure should be like : {}".format(folder_structure)
    assert os.path.exists(office1000_parent_folder), warning2

    for sub1_folder in sub1_folders:
        for sub2_folder in sub2_folders:
            for sub3_folder in sub3_folders:
                for sub4_folder in sub4_folders:
                    path = os.path.join(get_project_path(), "data", sub1_folder, sub2_folder, sub3_folder, sub4_folder)
                    warning3 = "\n" + path
                    assert os.path.exists(path), warning2 + warning3
                    if sub3_folder == "train":
                        indexes = [i for i in range(1200)]
                        for i in indexes:
                            file_path = os.path.join(path, "env_{}.pkl".format(indexes[i]))
                            if not os.path.exists(file_path):
                                print("Path not exist:{}".format(file_path))
                    else:
                        indexes = [i for i in range(240)]
                        for i in indexes:
                            file_path = os.path.join(path, "env_{}.pkl".format(indexes[i]))
                            if not os.path.exists(file_path):
                                print("Path not exist:{}".format(file_path))