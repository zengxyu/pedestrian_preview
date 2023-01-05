#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/14/22 2:51 PM 
    @Description    :
        
===========================================
"""
import glob
import itertools
import json
import os.path
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def read_json(path):
    print(os.listdir(path))
    # 读取所有metric.json文件的路径 read all paths of metric.json in given folder
    metric_json_paths = glob.glob(path + "/**/*.json", recursive=True)
    # 读取所有的metric.json文件，并保存在此list中, read content from all metric.json files
    metric_jsons = []
    for path in metric_json_paths:
        f = open(path, 'r')
        metric_json = json.load(f)
        metric_jsons.append(metric_json)
    return metric_jsons


def iterate_conditions(conditions: Dict):
    values_keys_lists = []
    for key, values in conditions.items():
        value_key_list = []
        for value in values:
            value_key_list.append({value: key})
        values_keys_lists.append(value_key_list)
    # unfold conditions
    res = list(itertools.product(*values_keys_lists))

    # invert {value:key} to {key:value}
    new_values_keys_lists = []
    for value_key_list in res:
        new_value_key_list = {}
        for item in value_key_list:
            for value, key in item.items():
                new_value_key_list[key] = value
        new_values_keys_lists.append(new_value_key_list)
    return new_values_keys_lists


def check_meet_all_conditions(conditions, metric_json):
    """
    check if this metric json meet all conditions
    :param conditions:
    :param metric_json:
    :return:
    """
    meet = True
    for key in conditions:
        try:
            if conditions[key] != metric_json[key]:
                meet = False
        except:
            print("key:{}".format(key))

    return meet


def extract_metric(metric_jsons, conditions: Dict, metric_name):
    conditions_list: List[Dict] = iterate_conditions(conditions)
    condition_values = []
    filtered_metric_jsons = []
    for conditions_dict in conditions_list:
        for metric_json in metric_jsons:
            is_meet_conditions = check_meet_all_conditions(conditions_dict, metric_json)
            if is_meet_conditions:
                condition_values.append(metric_json[metric_name])
                filtered_metric_jsons.append(metric_json)
    return condition_values