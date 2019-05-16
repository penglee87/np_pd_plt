import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

X = np.array([[-1,-1],[-2,-1],[1,1],[2,1],[-1,1],[-1,2],[1,-1],[1,-2]])
y = np.array([0,0,1,1,2,2,3,3])
# y=np.array([1,1,2,2,3,3,4,4])

#SVC 参数说明
#kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
#kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
#decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
#decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。


# clf = SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
# clf = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
# clf = SVC(decision_function_shape="ovr",probability=True)
clf = SVC(probability=True)
clf.fit(X, y)
'''
对于n分类，会有n个分类器，然后，任意两个分类器都可以算出一个分类界面，这样，用decision_function()时，对于任意一个样例，就会有n*(n-1)/2个值。
任意两个分类器可以算出一个分类界面，然后这个值就是距离分类界面的距离。
我想，这个函数是为了统计画图，对于二分类时最明显，用来统计每个点离超平面有多远，为了在空间中直观的表示数据以及画超平面还有间隔平面等。
decision_function_shape="ovr"时是4个值，为ovo时是6个值。
'''
print('decision_function',clf.decision_function(X))  #decision_function中每一列的值代表距离各类别的距离。
print('predict',clf.predict(X))  #预测分类
print('predict_proba',clf.predict_proba(X))  #这个是得分,每个分类器的得分，取最大得分对应的类。
#画图
plot_step=0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
 
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) #对坐标风格上的点进行预测，来画分界面。其实最终看到的类的分界线就是分界面的边界线。
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")
 
class_names="ABCD"
plot_colors="rybg"
for i, n, c in zip(range(4), class_names, plot_colors):
    idx = np.where(y == i) #i为0或者1，两个类
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')
plt.show()
