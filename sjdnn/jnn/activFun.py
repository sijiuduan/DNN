import numpy as np
from jnn.ea import Ea

def linear(z):
    return z
def dLinear(z):
    return 1

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
def dSigmoid(z):
    s = sigmoid(z)
    dz = s * (1-s)
    return dz

def tanh(z):
    return np.tanh(z)
def dTanh(z):
    t = tanh(z)
    dz = 1-t**2
    return dz

def relu(z):
    g = np.fmax(z, 0)
    return g
def dRelu(z):
    g = np.fmax(z,0)
    g = np.sign(g)
    return g

def leakyRelu(z):
    g = np.fmax(z, 0.01*z)
    return g
def dLeakyRelu(z):
    g = np.piecewise(z, [z < 0, z > 0], [.01, 1])
    return g

def softmax(z):
    x_exp = np.exp(z)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / x_sum
    return s

def dSoftmax(z):
    # 你需要指定 dCrossSoftmax 为损失导函数
    return 1

eaActivFun = Ea()
eaActivFun.linear.g = linear
eaActivFun.linear.dg = dLinear
eaActivFun.linear.name = 'linear'

eaActivFun.sigmoid.g = sigmoid
eaActivFun.sigmoid.dg = dSigmoid
eaActivFun.sigmoid.name = 'sigmoid'

eaActivFun.tanh.g = tanh
eaActivFun.tanh.dg = dTanh
eaActivFun.tanh.name = 'tanh'

eaActivFun.relu.g = relu
eaActivFun.relu.dg = dRelu
eaActivFun.relu.name = 'relu'

eaActivFun.leakyRelu.g = leakyRelu
eaActivFun.leakyRelu.dg = dLeakyRelu
eaActivFun.leakyRelu.name = 'leakyRelu'

eaActivFun.softmax.g = softmax
eaActivFun.softmax.dg = dSoftmax
eaActivFun.softmax.name = 'softmax'

if __name__ == '__main__':
    Ea.show(eaActivFun)
