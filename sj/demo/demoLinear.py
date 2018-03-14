from jnn.plot import *
from jnn.nn import *
from jnn.dataFactory import *
from jnn.activFun import eaActivFun
from jnn.costFun import eaCostFun
from jnn.ea import Ea

def eaNetLinear(eaData,L=4,n=4):
    net = Ea()
    net.L = L
    net.n[0] = eaData.X.shape[0]  # nx
    for l in range(1,L):
        net.n[l]=n
        net.activFun[l] = eaActivFun.linear
    net.n[L] = 1
    net.activFun[L] = eaActivFun.linear

    net.costFun = eaCostFun.L2
    return net

def demoLinear():
    eaData = eaDataLinear([2, -3.4], [4.2], m=1000)
    net = eaNetLinear(eaData,L=1,n=4)
    nnInitWb(net)
    nnFit(eaData, net, learn_rate=0.02)
    Ea.show(net, 3)
    plotCost("Linear Costs", net.costs,netInfo=eaNetInfo(net))

if __name__ == '__main__':
    demoLinear()

