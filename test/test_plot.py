import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.05, 10, 1000)
y = np.sin(x)
plt.plot(x, y, label="plot figure", ls=":", c="red", lw=2)
plt.legend()
# xy被注释图形内容位置坐标，xytext注释文本的位置坐标，color注释文本的颜色。arrowprops指示被注释内容的箭头的属性字典
plt.annotate("maxmin", xy=(np.pi / 2, 1.0), xytext=((np.pi / 2) + 1, 0.8), color="blue", weight="bold",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
# 添加无指示标记文本注释
plt.text(4.10, 0.09, "y=sin(x)", weight="bold", color="green")
plt.xlabel("x-axis", fontsize=15)
plt.ylabel("y-axis", fontsize=15)
plt.title("y=sin(x)", fontsize=15)
plt.show()
