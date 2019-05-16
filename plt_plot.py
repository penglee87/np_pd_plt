import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5.0, 5.0, 0.1)

y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

plt.figure(1)

#创建3行1列的子图，画其中第1个图
plt.subplot(311)
plt.plot(x, y1)

#创建3行1列的子图，画其中第2个图
plt.subplot(312)
plt.plot(x, y2,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=1,alpha=1,markeredgewidth=0.3,markeredgecolor="grey")  #返回一个 Line2D表示绘图数据的对象列表。

#创建3行1列的子图，画其中第3个图
plt.subplot(313)
plt.xlim((-5, 5))
plt.ylim((-10, 10))  #更改坐标轴取值范围
plt.plot(x, y3)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

z = np.arange(-10,10,0.1)
p = sigmoid(z)
plt.plot(z,p)
#画一条竖直线，如果不设定x的值，则默认是0
plt.axvline(x=0, color='k')
plt.axhspan(0.0, 1.0,facecolor='0.7',alpha=0.4)
# 画一条水平线，如果不设定y的值，则默认是0
plt.axhline(y=1, ls='dotted', color='0.4')
plt.axhline(y=0, ls='dotted', color='0.4')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.ylim(-0.1,1.1)
#确定y轴的坐标
plt.yticks([0.0, 0.5, 1.0])
plt.ylabel('$\phi (z)$')
plt.xlabel('z')
ax = plt.gca()
ax.grid(True)
plt.show()


'''
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.plot(x, y1)

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
ax.plot(x, y2)

fig3 = plt.figure(3)
ax = fig3.add_subplot(211)
ax.plot(x, y1)
ax = fig3.add_subplot(212)
ax.plot(x, y2)

plt.show()
'''