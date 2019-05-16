import numpy as np
from matplotlib import pyplot as plt


#等高线图


def f(x,y):
    return (1 - x/2 + x**5 + y**3) * np.exp(-x**2,-y**2)

n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)
X,Y=np.meshgrid(x,y)

#plt.contourf把颜色加进去，位置参数分别为:X，Y，f(X,Y)。透明度0.75，并将f(X,Y)的值对应到color map的暖色组中寻找对应颜色
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)
#plt.contour函数划线。位置参数为 X，Y，f(X,Y)。将颜色选为黑色，线条宽度为0.5
C=plt.contour(X,Y,f(X,Y),8,colors='black',linewidth=0.5)
#添加高度数字。我们加入Label,inline控制是否在Label画在线里面，字体大小为10.并将坐标轴隐藏：
plt.clabel(C,inline=True,fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()
plt.show()