from jnn.activFun import *
from jnn.costFun import *
from jnn.dataFactory import *

def eaNetInfo(net):
    info = Ea()
    info.X.n = net.n[0]

    for l in range(1, net.L):
        k = "layer %d" % l
        info[k].n = net.n[l]
        info[k].activFun = net.activFun[l].name

    info.Y.n = net.n[net.L]
    info.Y.activFun = net.activFun[net.L].name
    info.Y.costFun = net.costFun.name
    info.learn_rate = net.learn_rate
    # Ea.show(info)
    return info


# 判断梯度截断 TruncationGradient (TG)参数
def isTruncationGradient(costs):
    latest = 0.2
    c = int(len(costs) * latest)
    if c < 2:
        return False
    c1 = np.mean(costs[-c:-1])
    c2 = np.mean(costs[-c - c: -1])

    if c2/costs[0] < 0.8:
        return np.std([c1, c2])/costs[0] < 0.001
    elif c2/costs[0] > 1.5:
        return True
    else:
        return False

# DNN 网络 随机初始化 W, b
def nnInitWb(net):
    L = net.L
    n = net.n

    p = 0.2
    for l in range(1, L + 1):
        np.random.seed(2)
        net.W[l] = np.random.normal(size=(n[l], n[l - 1])) * p
        net.b[l] = np.zeros(shape=(n[l], 1))
    return net

# DNN 网络 根据网络模型，预测结果
def nnPredict(net, X):
    L = net.L
    W = net.W
    b = net.b
    A = Ea()
    A[0] = X
    Z = Ea()

    for l in range(1, L + 1):
        Z[l] = np.dot(W[l], A[l - 1]) + b[l]
        A[l] = net.activFun[l].g(Z[l])

    if net.activFun[l].name == "sigmoid":
        y_hat = A[L]
        y_hat = np.int32(np.array(0.5 < y_hat))
    elif net.activFun[l].name == "softmax":
        y_hat = A[L]
        y_hat = np.int32(np.array(y_hat > 0.5))
        for i in range(0,y_hat.shape[0]):
            y_hat[i] = i*y_hat[i]
        y_hat = np.sum(y_hat, axis=0)
    else:
        y_hat = A[L]

    return y_hat

# DNN 网络 迭代
def nnFit(eaData, net, learn_rate=0.02, iteration_num=10000):
    net.learn_rate = learn_rate
    net.costs = []

    X = eaData.X
    Y = eaData.Y
    A=Ea()
    A[0]=X
    dA = Ea()
    dZ = Ea()
    dW = Ea()
    db = Ea()
    Z=Ea()

    for i in range(1, iteration_num):
        # forward properation
        for l in range(1, net.L + 1):
            Z[l] = np.dot(net.W[l], A[l - 1]) + net.b[l]
            A[l] = net.activFun[l].g(Z[l])

        cost = net.costFun.J(A[net.L], Y)
        net.costs.append(cost)
        dA[net.L] = net.costFun.dJ(A[net.L], Y)

        # back properation
        for l in range(0, net.L):
            l = net.L - l
            _dg = net.activFun[l].dg(Z[l])
            dZ[l] = _dg * dA[l]
            dA[l - 1] = np.dot(net.W[l].T, dZ[l])
            dW[l] = 1 / X.shape[1] * np.dot(dZ[l], A[l - 1].T)
            db[l] = 1 / X.shape[1] * np.sum(dZ[l], axis=1, keepdims=True)

        # update parameters
        for l in range(0, net.L):
            l = net.L - l
            net.W[l] = net.W[l] - learn_rate * dW[l]
            net.b[l] = net.b[l] - learn_rate * db[l]

        if isTruncationGradient(net.costs):
            break

    return net

