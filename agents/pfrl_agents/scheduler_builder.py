#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 8/10/22 3:33 PM 
    @Description    :
        
===========================================
"""
from torch.optim.lr_scheduler import StepLR, _LRScheduler

from utils.config_utility import read_yaml


def get_scheduler(parser_args, name, optimizer) -> _LRScheduler:
    config = read_yaml(config_dir=parser_args.agents_config_folder, config_name="scheduler.yaml")
    scheduler = StepLR(optimizer, step_size=config[name]["step_size"], gamma=config[name]["gamma"], last_epoch=-1)

    return scheduler


class SchedulerHandler:
    def __init__(self, parser_args, name, optimizers):
        self.adjust_history = {"0.4": False, "0.6": False, "0.8": False, "0.95": False}
        self.optimizers = optimizers
        self.schedulers = []
        for optimizer in optimizers:
            scheduler = get_scheduler(parser_args, name, optimizer)
            self.schedulers.append(scheduler)

    def _step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def print(self):
        for optimizer in self.optimizers:
            print("{}: lr: {}".format(optimizer.__class__, optimizer.state_dict()['param_groups'][0]['lr']))

    def lr_schedule(self, success_rate):
        """
        当成功率大于某个值得时候，调整学习率, 调整过后除非到达下一个成功率，否则不再调整
        :param success_rate:
        :return:
        """
        for sc in self.adjust_history:
            sc_float = float(sc)
            if not self.adjust_history[sc] and success_rate > sc_float:
                self._step()
                self.print()
                self.adjust_history[sc] = True
