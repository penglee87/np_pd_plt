
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap


# 分层k折交叉验证示意图
def plot_stratified_cross_validation():
    fig, both_axes = plt.subplots(2, 1, figsize=(12, 5))
    # plt.title("cross_validation_not_stratified")
    axes = both_axes[0]
    axes.set_title("Standard cross-validation with sorted class labels")

    axes.set_frame_on(False)

    n_folds = 3
    n_samples = 150

    n_samples_per_fold = n_samples / float(n_folds)

    for i in range(n_folds):
        colors = ["w"] * n_folds
        colors[i] = "grey"
        axes.barh(y=range(n_folds), width=[n_samples_per_fold - 1] *
                  n_folds, left=i * n_samples_per_fold, height=.6,
                  color=colors, hatch="//", edgecolor='k', align='edge')

    axes.barh(y=[n_folds] * n_folds, width=[n_samples_per_fold - 1] *
              n_folds, left=np.arange(3) * n_samples_per_fold, height=.6,
              color="w", edgecolor='k', align='edge')

    axes.invert_yaxis()
    axes.set_xlim(0, n_samples + 1)
    axes.set_ylabel("CV iterations")
    axes.set_xlabel("Data points")
    axes.set_xticks(np.arange(n_samples_per_fold / 2.,
                              n_samples, n_samples_per_fold))
    axes.set_xticklabels(["Fold %d" % x for x in range(1, n_folds + 1)])
    axes.set_yticks(np.arange(n_folds + 1) + .3)
    axes.set_yticklabels(
        ["Split %d" % x for x in range(1, n_folds + 1)] + ["Class label"])
    for i in range(3):
        axes.text((i + .5) * n_samples_per_fold, 3.5, "Class %d" %
                  i, horizontalalignment="center")

    ax = both_axes[1]
    ax.set_title("Stratified Cross-validation")
    ax.set_frame_on(False)
    ax.invert_yaxis()
    ax.set_xlim(0, n_samples + 1)
    ax.set_ylabel("CV iterations")
    ax.set_xlabel("Data points")

    ax.set_yticks(np.arange(n_folds + 1) + .3)
    ax.set_yticklabels(
        ["Split %d" % x for x in range(1, n_folds + 1)] + ["Class label"])

    n_subsplit = n_samples_per_fold / 3.
    for i in range(n_folds):
        test_bars = ax.barh(
            y=[i] * n_folds, width=[n_subsplit - 1] * n_folds,
            left=np.arange(n_folds) * n_samples_per_fold + i * n_subsplit,
            height=.6, color="grey", hatch="//", edgecolor='k', align='edge')

    w = 2 * n_subsplit - 1
    ax.barh(y=[0] * n_folds, width=[w] * n_folds, left=np.arange(n_folds)
            * n_samples_per_fold + (0 + 1) * n_subsplit, height=.6, color="w",
            hatch="//", edgecolor='k', align='edge')
    ax.barh(y=[1] * (n_folds + 1), width=[w / 2., w, w, w / 2.],
            left=np.maximum(0, np.arange(n_folds + 1) * n_samples_per_fold -
                            n_subsplit), height=.6, color="w", hatch="//",
            edgecolor='k', align='edge')
    training_bars = ax.barh(y=[2] * n_folds, width=[w] * n_folds,
                            left=np.arange(n_folds) * n_samples_per_fold,
                            height=.6, color="w", hatch="//", edgecolor='k',
                            align='edge')

    ax.barh(y=[n_folds] * n_folds, width=[n_samples_per_fold - 1] *
            n_folds, left=np.arange(n_folds) * n_samples_per_fold, height=.6,
            color="w", edgecolor='k', align='edge')

    for i in range(3):
        ax.text((i + .5) * n_samples_per_fold, 3.5, "Class %d" %
                i, horizontalalignment="center")
    ax.set_ylim(4, -0.1)
    plt.legend([training_bars[0], test_bars[0]], [
               'Training data', 'Test data'], loc=(1.05, 1), frameon=False)

    fig.tight_layout()
    
    
plot_stratified_cross_validation()
plt.show()