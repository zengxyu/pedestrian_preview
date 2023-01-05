#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 12/5/22 11:52 PM 
    @Description    :
        
===========================================
"""
import logging
import os

import numpy as np
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from PIL import Image
import cv2

log = logging.getLogger(__name__)


class MyInteractiveIndoorScene(InteractiveIndoorScene):
    def __init__(self, **kwargs):
        super(MyInteractiveIndoorScene, self).__init__(self, **kwargs)

    def load_trav_map(self, maps_path):
        """
        Loads the traversability maps for all floors

        :param maps_path: String with the path to the folder containing the traversability maps
        """
        if not os.path.exists(maps_path):
            log.warning("trav map does not exist: {}".format(maps_path))
            return

        self.floor_map = []
        self.floor_graph = []
        for floor in range(len(self.floor_heights)):
            if self.trav_map_type == "with_obj":
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_door_{}.png".format(floor))))
            else:
                trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor))))

            # If we do not initialize the original size of the traversability map, we obtain it from the image
            # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
            # original map size
            if self.trav_map_original_size is None:
                height, width = trav_map.shape
                assert height == width, "trav map is not a square"
                self.trav_map_original_size = height
                self.trav_map_size = int(
                    self.trav_map_original_size * self.trav_map_default_resolution / self.trav_map_resolution
                )

            # We resize the traversability map to the new size computed before
            trav_map = cv2.resize(trav_map, (self.trav_map_size, self.trav_map_size))

            # We then erode the image. This is needed because the code that computes shortest path uses the global map
            # and a point robot
            if self.trav_map_erosion != 0:
                trav_map = cv2.erode(trav_map, np.ones((self.trav_map_erosion, self.trav_map_erosion)))
            import matplotlib.pyplot as plt
            plt.imshow(trav_map)
            plt.show()
            # We make the pixels of the image to be either 0 or 255
            trav_map[trav_map < 255] = 0

            # We search for the largest connected areas
            if self.build_graph:
                self.build_trav_graph(maps_path, floor, trav_map)
            self.floor_map.append(trav_map)
