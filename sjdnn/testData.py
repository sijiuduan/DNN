import numpy as np
import costFunctions as cfn
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt

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
    np.random.seed(1)
    X = 2*np.random.randn(2,m)
    R = np.linalg.norm(X,ord=2,axis=0).reshape(1,m)
    np.random.seed(2)
    noise = .3 * np.random.normal(size=R.shape)
    R = R + noise
    R = np.int32(np.array(R < 2.5))
    X = np.r_[X,R]
    f = X[:,X[2,:]==0]
    t = X[:,X[2,:]==1]
    f[2]=0
    t[2]=1
    return t,f

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

def plotGR(g,r,alpha=0.2):
    # plt.title("Red @ Blue")
    plt.grid(True)
    plt.plot(r[0], r[1], 'ro', alpha=alpha, label='red')
    plt.plot(g[0], g[1], 'go', alpha=alpha, label='blue')
    plt.legend()
    plt.show()

# t,f = tf_data()
# plotGR(t,f)
# print(t.shape,f.shape, np.c_[t,f][2:])



