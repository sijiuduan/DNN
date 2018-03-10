
import matplotlib.pyplot as plt
from jnn.ea import Ea
import numpy as np
from matplotlib.colors import ListedColormap
from jnn.nn import *

def plotCost(title,costs,netInfo):
    plt.figure(figsize=(10,6),dpi=100,facecolor='w')
    plt.grid(True)
    plt.title(title)
    iter_count = len(costs)
    plt.plot(range(0, iter_count), costs, label='J (A,Y)')
    if isinstance(netInfo,Ea):
        plt.text(iter_count*0.74, costs[0]*1.02, netInfo.to_str("Net Info"), va='top',ha='left',bbox=dict(boxstyle='round,pad=0.1', fc='yellow', ec='k',lw=1 ,alpha=0.5))

    # plt.xlim(-iter_count/1.9, iter_count * 1.1)
    # plt.ylim(0, costs[0] * 1.1)

    plt.ylabel("Cost", fontsize=14)
    plt.xlabel("Iterators", fontsize=14)
    plt.text(iter_count * .8, (costs[0]-costs[-1]) * .01 + costs[-1], "Cost:%f" % costs[-1])
    # plt.legend()
    plt.show()

def plotPredict(eaNet, X, yIndex, cmap=['#0099CC', '#FF6666','#6622FF','#66FF66']):
    # get x_show
    def extend(a, b):
        return 1.05 * a - 0.05 * b, 1.05 * b - 0.05 * a

    x1_min, x1_max = extend(X[0].min(), X[0].max())  # x1的范围
    x2_min, x2_max = extend(X[1].min(), X[1].max())  # x2的范围

    N = 190
    M = 190
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=0)

    # predict x_show
    y_hat = nnPredict(eaNet, x_show)
    y_hat = y_hat.reshape(x1.shape)

    plt.figure(figsize=(8, 8), dpi=100, facecolor='w')
    cm_light = ListedColormap(cmap)
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light, alpha=0.5)

    yIndex = np.squeeze(yIndex)
    plt.scatter(X[0], X[1], s=20, c=yIndex, edgecolors='k', cmap=cm_light)

    plt.show()