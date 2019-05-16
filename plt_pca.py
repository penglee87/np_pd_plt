from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from sklearn.externals.joblib import Memory

memory = Memory(cachedir="cache")
'''
主成分分析
'''

# PCA 对一个模拟二维数据集的作用
def plot_pca_illustration():
    rnd = np.random.RandomState(5)
    X_ = rnd.normal(size=(300, 2))
    X_blob = np.dot(X_, rnd.normal(size=(2, 2))) + rnd.normal(size=2)  #np.dot 矩阵积

    pca = PCA()
    pca.fit(X_blob)
    X_pca = pca.transform(X_blob)

    S = X_pca.std(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    axes[0].set_title("Original data")
    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=X_pca[:, 0], linewidths=0,
                    s=60, cmap='viridis')
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].arrow(pca.mean_[0], pca.mean_[1], S[0] * pca.components_[0, 0],
                  S[0] * pca.components_[0, 1], width=.1, head_width=.3,
                  color='k')
    axes[0].arrow(pca.mean_[0], pca.mean_[1], S[1] * pca.components_[1, 0],
                  S[1] * pca.components_[1, 1], width=.1, head_width=.3,
                  color='k')
    axes[0].text(-1.5, -.5, "Component 2", size=14)
    axes[0].text(-4, -4, "Component 1", size=14)
    axes[0].set_aspect('equal')  #设置轴缩放的纵横比，即Y轴单位与X轴单位的比率。('equal','auto',num 具体比例)

    axes[1].set_title("Transformed data")
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=X_pca[:, 0], linewidths=0,
                    s=60, cmap='viridis')
    axes[1].set_xlabel("First principal component")
    axes[1].set_ylabel("Second principal component")
    axes[1].set_aspect('equal')
    axes[1].set_ylim(-8, 8)

    pca = PCA(n_components=1)  #n_components 要保留的主成分个数，n_components='mle'为自动选取特征个数，使得满足所要求的方差百分比。
    pca.fit(X_blob)
    X_inverse = pca.inverse_transform(pca.transform(X_blob))  #pca.inverse_transform 将降维后的数据转换成原始数据

    axes[2].set_title("Transformed data w/ second component dropped")
    axes[2].scatter(X_pca[:, 0], np.zeros(X_pca.shape[0]), c=X_pca[:, 0],
                    linewidths=0, s=60, cmap='viridis')
    axes[2].set_xlabel("First principal component")
    axes[2].set_aspect('equal')
    axes[2].set_ylim(-8, 8)

    axes[3].set_title("Back-rotation using only first component")
    axes[3].scatter(X_inverse[:, 0], X_inverse[:, 1], c=X_pca[:, 0],
                    linewidths=0, s=60, cmap='viridis')
    axes[3].set_xlabel("feature 1")
    axes[3].set_ylabel("feature 2")
    axes[3].set_aspect('equal')
    axes[3].set_xlim(-8, 4)
    axes[3].set_ylim(-8, 4)


# 启用 PCA 的白化（whitening）选项，它将主成分缩放到相同的尺度。变换后的结果与使用 StandardScaler 相同。白化不仅对应于旋转数据，还对应于缩放数据使其形状是圆形而不是椭圆
def plot_pca_whitening():
    rnd = np.random.RandomState(5)
    X_ = rnd.normal(size=(300, 2))
    X_blob = np.dot(X_, rnd.normal(size=(2, 2))) + rnd.normal(size=2)

    pca = PCA(whiten=True)
    pca.fit(X_blob)
    X_pca = pca.transform(X_blob)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes = axes.ravel()

    axes[0].set_title("Original data")
    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=X_pca[:, 0], linewidths=0, s=60, cmap='viridis')
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].set_aspect('equal')

    axes[1].set_title("Whitened data")
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=X_pca[:, 0], linewidths=0, s=60, cmap='viridis')
    axes[1].set_xlabel("First principal component")
    axes[1].set_ylabel("Second principal component")
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-3, 4)


#@memory.cache
#def pca_faces(X_train, X_test):
#    # copy and pasted from nmf. refactor?
#    # Build NMF models with 10, 50, 100, 500 components
#    # this list will hold the back-transformd test-data
#    reduced_images = []
#    for n_components in [10, 50, 100, 500]:
#        # build the NMF model
#        pca = PCA(n_components=n_components)
#        pca.fit(X_train)
#        # transform the test data (afterwards has n_components many dimensions)
#        X_test_pca = pca.transform(X_test)
#        # back-transform the transformed test-data
#        # (afterwards it's in the original space again)
#        X_test_back = pca.inverse_transform(X_test_pca)
#        reduced_images.append(X_test_back)
#    return reduced_images
#
#
#def plot_pca_faces(X_train, X_test, image_shape):
#    reduced_images = pca_faces(X_train, X_test)
#
#    # plot the first three images in the test set:
#    fix, axes = plt.subplots(3, 5, figsize=(15, 12),
#                             subplot_kw={'xticks': (), 'yticks': ()})
#    for i, ax in enumerate(axes):
#        # plot original image
#        ax[0].imshow(X_test[i].reshape(image_shape),
#                     vmin=0, vmax=1)
#        # plot the four back-transformed images
#        for a, X_test_back in zip(ax[1:], reduced_images):
#            a.imshow(X_test_back[i].reshape(image_shape), vmin=0, vmax=1)
#
#    # label the top row
#    axes[0, 0].set_title("original image")
#    for ax, n_components in zip(axes[0, 1:], [10, 50, 100, 500]):
#        ax.set_title("%d components" % n_components)

        
#plot_pca_illustration()
plot_pca_whitening()
plt.show()