import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

mpl.rcParams['legend.fontsize'] = 20  # mpl模块载入的时候加载配置信息存储在rcParams变量中，rc_params_from_file()函数从文件加载配置信息

font = {
    'color': 'b',
    'style': 'oblique',
    'size': 20,
    'weight': 'bold'
}

#Line plot (三维线图)
fig = plt.figure(figsize=(16, 12))  #参数为图片大小
ax = fig.gca(projection='3d')  # get current axes，且坐标轴是3d的
#ax = Axes3D(fig, elev=-152, azim=-26)  #Axes3D类用法  elev存储z平面中的仰角,azim存储x，y平面中的方位角

# 准备数据
theta = np.linspace(-8 * np.pi, 8 * np.pi, 100)  # 生成等差数列，[-8π,8π]，个数为100
z = np.linspace(-2, 2, 100)  # [-2,2]容量为100的等差数列，这里的数量必须与theta保持一致，因为下面要做对应元素的运算
r = z ** 2 + 1
x = r * np.sin(theta)  # [-5,5]
y = r * np.cos(theta)  # [-5,5]
ax.set_xlabel("X", fontdict=font)
ax.set_ylabel("Y", fontdict=font)
ax.set_zlabel("Z", fontdict=font)
ax.set_title("Line Plot", alpha=0.5, fontdict=font) #alpha参数指透明度transparent
ax.plot(x, y, z, label='parametric curve')
ax.legend(loc='upper right') #legend的位置可选：upper right/left/center,lower right/left/center,right,left,center,best等等

'''
#Scatter plot (三维散点图)
label_font = {
    'color': 'c',
    'size': 15,
    'weight': 'bold'
}
def randrange(n, vmin, vmax):
    r = np.random.rand(n)  # 随机生成n个介于0~1之间的数
    return (vmax - vmin) * r + vmin  # 得到n个[vmin,vmax]之间的随机数


fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection="3d")  # 添加子坐标轴，111表示1行1列的第一个子图
n = 200
for zlow, zhigh, c, m, l in [(4, 15, 'r', 'o', 'positive'),
                             (13, 40, 'g', '*', 'negative')]:  # 用两个tuple，是为了将形状和颜色区别开来
    x = randrange(n, 15, 40)
    y = randrange(n, -5, 25)
    z = randrange(n, zlow, zhigh)
    ax.scatter(x, y, z, c=c, marker=m, label=l, s=z * 10) #这里marker的尺寸和z的大小成正比

ax.set_xlabel("X axis", fontdict=label_font)
ax.set_ylabel("Y axis", fontdict=label_font)
ax.set_zlabel("Z axis", fontdict=label_font)
ax.set_title("Scatter plot", alpha=0.6, color="b", size=25, weight='bold', backgroundcolor="y")   #子图的title
ax.legend(loc="upper left")    #legend的位置左上


#Surface plot (三维曲面图)
fig = plt.figure(figsize=(16,12))
ax = fig.gca(projection="3d")

# 准备数据 
x = np.arange(-5, 5, 0.25)    #生成[-5,5]间隔0.25的数列，间隔越小，曲面越平滑
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x,y)  #格点矩阵,原来的x行向量向下复制len(y)次，形成len(y)*len(x)的矩阵，即为新的x矩阵；原来的y列向量向右复制len(x)次，形成len(y)*len(x)的矩阵，即为新的y矩阵；新的x矩阵和新的y矩阵shape相同
r = np.sqrt(x ** 2 + y ** 2)
z = np.sin(r)

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)  # cmap指color map

# 自定义z轴
ax.set_zlim(-1, 1)
ax.zaxis.set_major_locator(LinearLocator(20))  # z轴网格线的疏密，刻度的疏密，20表示刻度的个数
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  # 将z的value字符串转为float，保留2位小数

#设置坐标轴的label和标题
ax.set_xlabel('x',size=15)
ax.set_ylabel('y',size=15)
ax.set_zlabel('z',size=15)
ax.set_title("Surface plot", weight='bold', size=20)

#添加右侧的色卡条
fig.colorbar(surf, shrink=0.6, aspect=8)  # shrink表示整体收缩比例，aspect仅对bar的宽度有影响，aspect值越大，bar越窄
'''


plt.show()
