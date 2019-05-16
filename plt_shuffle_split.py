
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap


# 打乱划分交叉验证示意图
# 对包含 10 个点的数据集运行 4 次迭代划分，每次的训练集包含 5 个点，测试集包含 2 个点
def plot_shuffle_split():
    from sklearn.model_selection import ShuffleSplit
    plt.figure(figsize=(10, 2))
    plt.title("ShuffleSplit with 10 points"
              ", train_size=5, test_size=2, n_splits=4")

    axes = plt.gca()
    axes.set_frame_on(False)

    n_folds = 10
    n_samples = 10
    n_iter = 4
    n_samples_per_fold = 1

    ss = ShuffleSplit(n_splits=4, train_size=5, test_size=2, random_state=43)
    mask = np.zeros((n_iter, n_samples))
    for i, (train, test) in enumerate(ss.split(range(10))):
        mask[i, train] = 1
        mask[i, test] = 2

    for i in range(n_folds):
        # test is grey
        colors = ["grey" if x == 2 else "white" for x in mask[:, i]]
        # not selected has no hatch

        boxes = axes.barh(y=range(n_iter), width=[1 - 0.1] * n_iter,
                          left=i * n_samples_per_fold, height=.6, color=colors,
                          hatch="//", edgecolor='k', align='edge')
        for j in np.where(mask[:, i] == 0)[0]:
            boxes[j].set_hatch("")

    axes.invert_yaxis()
    axes.set_xlim(0, n_samples + 1)
    axes.set_ylabel("CV iterations")
    axes.set_xlabel("Data points")
    axes.set_xticks(np.arange(n_samples) + .5)
    axes.set_xticklabels(np.arange(1, n_samples + 1))
    axes.set_yticks(np.arange(n_iter) + .3)
    axes.set_yticklabels(["Split %d" % x for x in range(1, n_iter + 1)])
    # legend hacked for this random state
    plt.legend([boxes[1], boxes[0], boxes[2]], [
               "Training set", "Test set", "Not selected"], loc=(1, .3))
    plt.tight_layout()
    
    
plot_shuffle_split()
plt.show()