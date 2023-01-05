#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/16/22 5:30 PM 
    @Description    :
        
===========================================
"""
import numpy as np
from matplotlib import pyplot as plt, rcParams
from mpl_toolkits import axisartist


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# def softmax():
#     e ^ (x - max(x)) / sum(e ^ (x - max(x))
FONT_SIZE = 16


def plot_activation_function(x, y, title, yticks, save_path):
    rcParams.update({'figure.autolayout': True})

    plt.rc('font', size=FONT_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
    plt.grid()

    plt.plot(x, y, 'g')

    plt.ylabel(yticks)
    plt.title(title)

    plt.savefig(save_path)

    plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    x = np.arange(-5, 5, 0.1)
    y = tanh(x)
    plot_activation_function(x, y, "Tanh", "Tanh(x)", "plot/tanh.png")

    x = np.arange(-5, 5, 0.1)
    y = relu(x)
    plot_activation_function(x, y, "ReLU", "ReLU(x)", "plot/relu.png")

    # x = np.arange(-5, 5, 0.1)
    # y = tanh(x)
    # plot_activation_function(x, y, "Tanh", "Tanh(x)")
