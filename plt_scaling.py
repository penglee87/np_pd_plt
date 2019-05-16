import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, Normalizer,
                                   RobustScaler)
from matplotlib.colors import ListedColormap

'''
数据预处理与缩放,标准化的几种方法
'''
cm2 = ListedColormap(['#0000aa', '#ff2020'])
def plot_scaling():
    X, y = make_blobs(n_samples=50, centers=2, random_state=4, cluster_std=1)
    X += 3
    #标准化使用方法
    #StandardScaler().fit_transform(X)  #可用如下三行代替
    #scaler = StandardScaler()
    #scaler.fit(X)
    #X_scaled = scaler.transform(X)
    
    
    #MinMaxScaler().fit_transform(X)
    #Normalizer().fit_transform(X)
    #RobustScaler().fit_transform(X)

    plt.figure(figsize=(15, 8))
    main_ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)  #在2行4列的画布上，以(0,0)为起使位置，作一个大小为2行2列的图

    main_ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm2, s=60)
    maxx = np.abs(X[:, 0]).max()
    maxy = np.abs(X[:, 1]).max()

    main_ax.set_xlim(-maxx + 1, maxx + 1)
    main_ax.set_ylim(-maxy + 1, maxy + 1)
    main_ax.set_title("Original Data")
    other_axes = [plt.subplot2grid((2, 4), (i, j))
                  for j in range(2, 4) for i in range(2)]

    for ax, scaler in zip(other_axes, [StandardScaler(), RobustScaler(),
                                       MinMaxScaler(), Normalizer(norm='l2')]):
        X_ = scaler.fit_transform(X)
        ax.scatter(X_[:, 0], X_[:, 1], c=y, cmap=cm2, s=60)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(type(scaler).__name__)

    other_axes.append(main_ax)
    
    #定义x轴和y轴的位置和颜色
    for ax in other_axes:
        ax.spines['left'].set_position('center')  #将左坐标轴移至中间
        ax.spines['right'].set_color('none')  #将右坐标轴颜色置为空
        ax.spines['bottom'].set_position('center')  #将底部轴移至中间
        ax.spines['top'].set_color('none')  #将上部轴颜色置为空
        ax.xaxis.set_ticks_position('bottom')  #设置刻度显示的位置
        ax.yaxis.set_ticks_position('left')

        
plot_scaling()
plt.show()


