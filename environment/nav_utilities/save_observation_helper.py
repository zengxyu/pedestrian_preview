#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 9/5/22 3:10 PM 
    @Description    :
        
===========================================
"""
import logging
import os.path
import pickle


class ObservationSaver:
    cartesian_positions_list = []
    count = 0

    def store_observations(self, cartesian_positions_items):
        self.cartesian_positions_list.append(cartesian_positions_items)
        if len(self.cartesian_positions_list) >= 1000:
            folder = "output/observation_collections"
            if not os.path.exists(folder):
                os.makedirs(folder)
            file_path = os.path.join(folder, str(self.count + 1) + ".pkl")
            file = open(file_path, 'wb')
            # 2 x 3 x 2 x 180 x 2
            pickle.dump(self.cartesian_positions_list, file)
            self.cartesian_positions_list = []
            logging.info("------------------save data to {}. -----------------------".format(file_path))

        self.count += 1
