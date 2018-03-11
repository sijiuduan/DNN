from jnn.plot import *
from jnn.nn import *
from jnn.dataFactory import *
from jnn.activFun import eaActivFun
from jnn.costFun import eaCostFun
from jnn.ea import Ea

def eaNetLogistics(eaData):
    net = Ea()
    net.L = 2
    net.n[0] = eaData.X.shape[0]  # nx
    net.n[1] = 20
    net.n[2] = 4
    net.n[3] = 4
    net.n[net.L] = 1

    net.activFun[1] = eaActivFun.relu
    net.activFun[2] = eaActivFun.relu
    net.activFun[3] = eaActivFun.sigmoid
    net.activFun[net.L] = eaActivFun.sigmoid

    net.costFun = eaCostFun.L3
    return net

def demoLogistics():
    eaData = eaDataTFRing()
    net = eaNetLogistics(eaData)

    nnInitWb(net)
    nnFit(eaData, net, learn_rate=0.1)

    Ea.show(net, 3)

    plotPredict(net, eaData.X, eaData.Y, cmap=['#0099CC', '#FF6666','#6622FF'])
    plotCost("Logistics Costs", net.costs,netInfo=eaNetInfo(net))


if __name__ == '__main__':
    demoLogistics()