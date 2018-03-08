import numpy as np
import costFunctions as cfn
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Data
def linear_data(w = [2, -3.4],b = [4.2], m = 1000):
    nx = len(w)
    np.random.seed(1)
    X = np.random.normal(size=(nx, m))
    Y = np.dot(w, X) + b
    noise = .01 * np.random.normal(size=Y.shape)
    print("noise",cfn.L2(noise,0))
    Y += noise
    return X, Y


def tf_data(m=1000):
    #   np.random.seed(const_number)  保证每次输出结果不变。
    np.random.seed(1)
    X = 2 * np.random.randn(2, m)
    #   很方便的求 2-范数 的方法（相当于求半径）
    R = np.linalg.norm(X, ord=2, axis=0).reshape(1, m)
    oY = np.int32(np.array(R < 2.5))
    np.random.seed(2)
    #   增加一个噪音
    noise = .3 * np.random.normal(size=R.shape)
    R = R + noise
    #   二值化输出
    Y = np.int32(np.array(R < 2.5))

    result = np.int32(Y ^ oY)
    print("随机噪音造成错误的数据有%d条，占总数据的 %.2f%%" % (np.sum(result), 100 * np.sum(result) / m))

    #   合并输入输出数据 （学会用 np.r_[...] 和 np.c_[...])
    X = np.r_[X, Y]
    return X

def plot3d(x,y,z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title("Linear Data")
    # ax.plot(X[0],X[1],Y,'o')
    ax.scatter(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_tf(tf, alpha=0.2):
    f = tf[:, tf[-1, :] == 0]
    t = tf[:, tf[-1, :] == 1]

    plt.figure(figsize=(7, 7), dpi=56)
    plt.grid(True)
    plt.plot(f[0], f[1], 'ro', alpha=alpha, label='red')
    plt.plot(t[0], t[1], 'go', alpha=alpha, label='blue')
    plt.legend()
    plt.show()

    # x = tf
    # cm_light = ListedColormap(['g', 'r', 'b'])
    #
    # def extend(a, b):
    #     return 1.05 * a - 0.05 * b, 1.05 * b - 0.05 * a
    #
    # x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max())  # x1的范围
    # x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max())  # x2的范围
    #
    # N = 500; M = 500
    # t1 = np.linspace(x1_min, x1_max, N)
    # t2 = np.linspace(x2_min, x2_max, M)
    # x1, x2 = np.meshgrid(t1, t2)
    #
    # print(x1.shape)
    #
    # x_show = np.stack((x1.flat, x2.flat), axis=1)
    # y_hat = x[2]  # 预测
    # # y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同


def plotGR(g,r,alpha=0.2):
    # plt.title("Red @ Blue")
    plt.grid(True)
    plt.plot(r[0], r[1], 'ro', alpha=alpha, label='red')
    plt.plot(g[0], g[1], 'go', alpha=alpha, label='blue')
    plt.legend()
    plt.show()

# X = tf_data()
# plot_tf(X)
# print(t.shape,f.shape, np.c_[t,f][2:])



