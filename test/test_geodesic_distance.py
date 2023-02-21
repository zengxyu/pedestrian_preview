import os
import pytest
from utils.fo_utility import get_project_path


def test_geodesic_distance():
    parent_path = os.path.join(get_project_path(), "data", "office_1000", "geodesic_distance")
    for i in range(1000):
        path = os.path.join(parent_path, "env_{}.pkl".format(i))
        assert os.path.exists(path)
