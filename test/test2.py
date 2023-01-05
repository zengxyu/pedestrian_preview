import queue

import numpy as np
import matplotlib.pylab as plt

# def gaussian(x, mean=0, sigma=4):
#     a = 1
#     y = a * np.e ** (-(x - mean) ** 2 / (2 * sigma ** 2))
#     return y
#
#
# x = np.arange(-3, 3, 0.1)
# y = -abs(x) ** 2 * 0.01 + 1
# z = 0.8 ** abs(x)
# h = gaussian(x)
# plt.plot(x, y)
# plt.plot(x, z)
# plt.plot(x, h)
# plt.show()

# a = [5 / 42, 10 / 42, 15 / 42, 4 / 42, 8 / 42]
# # b = [0.5, 0.5]
#
# sum = 0
# for p in a:
#     sum -= p * np.log2(p)
# print(sum)
# print(np.log2(5))
# q = queue.Queue(2)
# q.put(1)
# q.put(2)
# q.put(3)
# print(q.queue)

# a = np.array([1.0, 1.0], dtype=np.int8)
# b = np.array([1, 1])
# print(a.dtype)
# print(b.dtype)
# b = b.astype(a.dtype)
# print(b.dtype)

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
b = a[::2]
c = a[1::2]
print(a)
print(b)
print(c)
