# -*- coding: utf-8 -*-
# http://blog.csdn.net/han_xiaoyang/article/details/49123419
# http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))

h = np.arange(-10, 10, 0.1) # 定义x的范围，像素为0.1
s_h = sigmoid(h) # sigmoid为上面定义的函数
plt.plot(h, s_h)
plt.axvline(0.0, color='k') # 在坐标轴上加一条竖直的线，0.0为竖直线在坐标轴上的位置
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted') # 加水平间距通过坐标轴
plt.axhline(y=0.5, ls='dotted', color='k') # 加水线通过坐标轴
plt.yticks([0.0, 0.5, 1.0]) # 加y轴刻度
plt.ylim(-0.1, 1.1) # 加y轴范围
plt.xlabel('h')
plt.ylabel('$S(h)$')
plt.show()
