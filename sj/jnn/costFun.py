import numpy as np
from jnn.ea import Ea

# 偏差代价函数
def bias(A, Y):
    # print(A.shape,Y.shape)
    loss = np.sum(np.abs(A-Y))
    return loss
# 偏差代价导函数
def dBias(A,Y):
    return np.sign(A-Y)
# 方差代价函数
def variance(A, Y):
    loss = 1/2 * np.dot((A-Y), (A-Y).T)
    loss = np.sum(loss)
    return loss
# 方差代价导函数
def dVariance(A,Y):
    return A-Y

# 交叉熵代价函数
def cross(A, Y):
    m = Y.shape[1]
    loss = -1 / m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A), axis=1, keepdims=False)
    return loss
# 交叉熵代价导函数
def dCross(A,Y):
    loss = -Y/A + (1-Y)/(1-A)
    # loss = np.squeeze(loss)
    return loss

# softmax 结果的交叉熵函数（对应多分类问题）
def crossSoftmax(A, Y):
    m = Y.shape[1]
    loss = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1, keepdims=False)
    loss = np.sum(loss)
    return loss

# softmax+交叉熵 结果的交叉熵导函数（对应多分类问题）
def dCrossSoftmax(A,Y):
    return A-Y

eaCostFun = Ea()
eaCostFun.L1.J = bias
eaCostFun.L1.dJ = dBias
eaCostFun.L1.name = 'bias'

eaCostFun.L2.J = variance
eaCostFun.L2.dJ = dVariance
eaCostFun.L2.name = 'variance'

eaCostFun.L3.J = cross
eaCostFun.L3.dJ = dCross
eaCostFun.L3.name = 'cross'

eaCostFun.L4.J = crossSoftmax
eaCostFun.L4.dJ = dCrossSoftmax
eaCostFun.L4.name = 'cross'

if __name__ == '__main__':
    Ea.show(eaCostFun)
