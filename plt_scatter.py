import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1,10)
y = x
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
ax1.scatter(x,y,c = 'r',marker = 'o'
            s=s,
            label=labels[i], alpha=alpha,
            linewidths =0.5,
            edgecolors ='grey')
plt.legend('x1')  #添加图例
plt.show() 