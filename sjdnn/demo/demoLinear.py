from jnn.plot import *
from jnn.nn import *
from jnn.activFun import eaActivFun
from jnn.costFun import eaCostFun
from jnn.dataFactory import *
from jnn.ea import Ea

def eaNetLinear(eaData):
    net = Ea()
    net.L = 1
    net.n[0] = eaData.X.shape[0]  # nx
    net.n[1] = 4
    net.n[2] = 4
    net.n[3] = 4
    net.n[4] = 1
    net.n[net.L] = 1

    net.activFun[1] = eaActivFun.linear
    net.activFun[2] = eaActivFun.linear
    net.activFun[3] = eaActivFun.linear
    net.activFun[4] = eaActivFun.linear

    net.costFun = eaCostFun.L2
    return net

def demoLinear():
    eaData = eaDataLinear([2, -3.4], [4.2], m=1000)
    net = eaNetLinear(eaData)
    net.L = 4

    nnInitWb(net)
    nnFit(eaData, net, learn_rate=0.02)

    Ea.show(net, 3)

    plotCost("Linear Costs", net.costs,netInfo=eaNetInfo(net))


if __name__ == '__main__':
    demoLinear()