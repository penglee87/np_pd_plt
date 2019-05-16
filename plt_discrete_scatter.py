import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs


#cm_cycle = ListedColormap(['#0000aa', '#ff5050', '#50ff50', '#9040a0', '#fff000'])
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
cm2 = ListedColormap(['#0000aa', '#ff2020'])


def make_forge():
    # a carefully hand-designed dataset lol
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)  #make_blobs 生成多类单标签数据集，默认含两个特征
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=np.bool)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y

# 画散点图
def discrete_scatter(x1, x2, y=None, markers=None, s=10, ax=None,
                     labels=None, padding=.2, alpha=1, c=None, markeredgewidth=None):
    """Adaption of matplotlib.pyplot.scatter to plot classes or clusters.

    Parameters
    ----------

    x1 : nd-array
        input data, first axis

    x2 : nd-array
        input data, second axis

    y : nd-array
        input data, discrete labels

    cmap : colormap
        Colormap to use.

    markers : list of string
        List of markers to use, or None (which defaults to 'o').

    s : int or float
        Size of the marker

    padding : float
        Fraction of the dataset range to use for padding the axes.

    alpha : float
        Alpha value for all points.
    """
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = np.zeros(len(x1))

    unique_y = np.unique(y)

    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8', '<', '>'] * 10

    if len(markers) == 1:
        markers = markers * len(unique_y)

    if labels is None:
        labels = unique_y

    # lines in the matplotlib sense, not actual lines
    lines = []

    current_cycler = mpl.rcParams['axes.prop_cycle']  #在rcparams中实现对prop-cycle属性markevery的支持
    #print('current_cycler',type(current_cycler))

    for i, (yy, cycle) in enumerate(zip(unique_y, current_cycler())):
        mask = y == yy
        # if c is none, use color cycle
        if c is None:
            color = cycle['color']
            #print('color',type(color),color)  #str类型
        elif len(c) > 1:
            color = c[i]
        else:
            color = c
        # use light edge for dark markers
        if np.mean(colorConverter.to_rgb(color)) < .4:
            markeredgecolor = "grey"
        else:
            markeredgecolor = "black"

        #plot返回一个 Line2D表示绘图数据的对象列表，plot()[0]返回具体的Line2D对象。
        lines.append(ax.plot(x1[mask], x2[mask], marker = markers[i], linestyle='', 
                             markersize=s,
                             label=labels[i], alpha=alpha, c=color,
                             markeredgewidth=markeredgewidth,
                             markeredgecolor=markeredgecolor)[0])
                             
        ##也可用scatter如下实现
        #lines.append(ax.scatter(x1[mask], x2[mask], marker = markers[i],
        #                     s=s*10,
        #                     label=labels[i], alpha=alpha, c=color,
        #                     linewidths =markeredgewidth,
        #                     edgecolors =markeredgecolor))

    if padding != 0:
        pad1 = x1.std() * padding
        pad2 = x2.std() * padding
        xlim = ax.get_xlim()  #返回一个元组，表示x轴值的范围
        #print('xlim',type(xlim),len(xlim),xlim[0],xlim[1])
        ylim = ax.get_ylim()
        ax.set_xlim(min(x1.min() - pad1, xlim[0]), max(x1.max() + pad1, xlim[1]))
        ax.set_ylim(min(x2.min() - pad2, ylim[0]), max(x2.max() + pad2, ylim[1]))

    return lines
    
    
#X1, y1 = make_blobs(centers=2, random_state=4, n_samples=30)  #make_blobs 生成多类单标签数据集，默认含两个特征
#X2, y2 = make_forge()
#
#
#plt.figure()
#discrete_scatter(X1[:, 0], X1[:, 1], y1)
#plt.legend( loc=4)
#plt.legend(["Class 0", "Class 1"], loc=1)  #loc=4 指定图例显示的位置
#plt.xlabel("First feature")
#plt.ylabel("Second feature")
#
#plt.figure()
#discrete_scatter(X2[:, 0], X2[:, 1], y2)


def plot_knn_classification(n_neighbors=1):
    X, y = make_forge()

    X_test = np.array([[8.2, 3.66214339], [9.9, 3.2], [11.2, .5]])
    print('X',type(X),X.shape)
    print('X_test',type(X_test),X_test.shape)
    dist = euclidean_distances(X, X_test)
    print('dist',type(dist),dist.shape,dist)
    closest = np.argsort(dist, axis=0)
    print('closest',type(closest),closest.shape,closest)

    for x, neighbors in zip(X_test, closest.T):
        #print('x',x,'neighbors',neighbors)
        for neighbor in neighbors[:n_neighbors]:
            plt.arrow(x[0], x[1], X[neighbor, 0] - x[0],
                      X[neighbor, 1] - x[1], head_width=0, fc='k', ec='k')

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    test_points = discrete_scatter(X_test[:, 0], X_test[:, 1], clf.predict(X_test), markers="*")
    training_points = discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(training_points + test_points, ["training class 0", "training class 1",
                                               "test pred 0", "test pred 1"])
                                               
                                               
#plot_knn_classification(n_neighbors=1)
plot_knn_classification(n_neighbors=3)


plt.show()