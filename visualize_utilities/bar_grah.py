#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/8/22 6:00 PM 
    @Description    :
        
===========================================
"""
import matplotlib.pyplot as plt
import numpy as np

FONT_SIZE = 14
x_names = ["office", "corridor", "cross"]
label_names = ["model1", "model2", "model3", "model4"]
values = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])


def auto_text(rects):
    for rect in rects:
        plt.text(rect.get_x() + 0.01, rect.get_height(), rect.get_height(), ha='left', va='bottom', fontsize=14)


markers = ['o', '*', 'h', 's', '1', '2', '3', '4']


def line_graph(x_names, label_names, values, colors, markers, title, x_label, y_label, save_path):
    values = np.around(values, 2)
    for i, label_name in enumerate(label_names):
        plt.plot(np.arange(0, len(values[i, :])), values[i, :], label=label_name, color=colors[i], alpha=0.75,
                 marker=markers[i],
                 ms=7)
        j = 0
        for x, y in zip(np.arange(0, len(values[i, :])), values[i, :]):
            """求在这一组数据中值大小拍第几"""
            a = abs(np.sort(values[:, j]) - y) <= 0.001
            y_float = np.argwhere(a)[0][0] - 1 - len(values[:, j]) / 2
            plt.text(x, y + 0.02 * y_float, y, ha='left', va='bottom', fontsize=10)
            j += 1

    plt.xticks(np.arange(0, len(x_names)), x_names, fontsize=FONT_SIZE)
    plt.ylim(0, np.max(values) + 0.05 * np.max(values))
    plt.yticks(fontsize=FONT_SIZE - 2)
    plt.legend(loc='lower right')
    # plt.title(title, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.xlabel(x_label, fontsize=FONT_SIZE)

    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=500)
    plt.show()


def bar_graph(x_names, label_names, values, colors, title, y_label, save_path, width=12, height=6.5, total_width=0.8
              , x_label=""):
    plt.figure(figsize=(width, 6.5))
    values = np.around(values, 2)
    width = total_width / len(label_names)
    n_x_names = len(x_names)
    left = np.arange(n_x_names) - (total_width - width) / 2

    bars = []
    for i, label_name in enumerate(label_names):
        bars.append(plt.bar(left + i * width, values[:, i], width=width, label=label_name, color=colors[i]))

    for bar in bars:
        auto_text(bar)

    plt.xticks(np.arange(0, len(x_names)), x_names, fontsize=FONT_SIZE)
    plt.ylim(0, np.max(values) + 0.1 * np.max(values))
    plt.yticks(fontsize=FONT_SIZE - 2)
    plt.legend(loc='lower right')
    # plt.title(title, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    if x_label != "":
        plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def bar_line_graph(x_names, label_names, values, colors, title, y_label, save_path, width=12, height=6.5, x_label=""):
    plt.figure(figsize=(width, 6.5))
    values = np.around(values, 2)
    total_width = 0.6
    width = total_width / len(label_names)
    n_x_names = len(x_names)
    left = np.arange(n_x_names) - (total_width - width) / 2

    bars = []
    for i, label_name in enumerate(label_names):
        bars.append(plt.bar(left + i * width, values[:, i], width=width, label=label_name, color=colors[i]))
        plt.plot(left + i * width, values[:, i], 'r')

    # for rects in bars:
    #     for rect in rects:
    #         plt.text(rect.get_x() + 0.01, rect.get_height(), rect.get_height(), ha='left', va='bottom', fontsize=14)

    for bar in bars:
        auto_text(bar)

    plt.xticks(np.arange(0, len(x_names)), x_names, fontsize=FONT_SIZE)
    plt.ylim(0, np.max(values) + 0.1 * np.max(values))
    plt.yticks(fontsize=FONT_SIZE - 2)
    plt.legend(loc='lower right')
    # plt.title(title, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    if x_label != "":
        plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def bar_graph_with_error(x_names, label_names, values, std_err1, colors, title, y_label, save_path, width=12,
                         height=6.5, total_width=0.8,
                         x_label=""):
    plt.figure(figsize=(width, 6.5))
    values = np.around(values, 2)
    width = total_width / len(label_names)
    n_x_names = len(x_names)
    left = np.arange(n_x_names) - (total_width - width) / 2

    bars = []
    for i, label_name in enumerate(label_names):
        bars.append(
            plt.bar(left + i * width, values[:, i], yerr=std_err1[:, i], width=width, label=label_name,
                    color=colors[i], ecolor='r'))

    for bar in bars:
        auto_text(bar)

    plt.xticks(np.arange(0, len(x_names)), x_names, fontsize=FONT_SIZE)
    plt.ylim(0, np.max(values) + 0.1 * np.max(values))
    plt.yticks(fontsize=FONT_SIZE - 2)
    plt.legend(loc='lower right')
    # plt.title(title, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    if x_label != "":
        plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def stack_bar_graph(x_names, label_names, values, colors, title, y_label, save_path, width=12, height=6.5, x_label=""):
    plt.title('Scores by group and gender')

    N = 13
    ind = np.arange(N)  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
    plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13'))

    plt.ylabel('Scores')
    plt.yticks(np.arange(0, 81, 20))

    Bottom = (52, 49, 48, 47, 44, 43, 41, 41, 40, 38, 36, 31, 29)
    Center = (38, 40, 45, 42, 48, 51, 53, 54, 57, 59, 57, 64, 62)
    Top = (10, 11, 7, 11, 8, 6, 6, 5, 3, 3, 7, 5, 9)

    d = []
    for i in range(0, len(Bottom)):
        sum = Bottom[i] + Center[i]
        d.append(sum)

    width = 0.6  # 设置条形图一个长条的宽度
    p1 = plt.bar(ind, Bottom, width, color='blue')
    p2 = plt.bar(ind, Center, width, bottom=Bottom, color='green')
    p3 = plt.bar(ind, Top, width, bottom=d, color='red')

    plt.legend((p1[0], p2[0], p3[0]), ('Bottom', 'Center', 'Top'), loc=3)

    plt.show()


if __name__ == '__main__':
    stack_bar_graph()
