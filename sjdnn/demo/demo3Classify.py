from jnn.plot import *
from jnn.nn import *
from jnn.activFun import eaActivFun
from jnn.costFun import eaCostFun
from jnn.dataFactory import *
from jnn.ea import Ea

def eaNetSoftmax(eaData):
    net = Ea()
    net.L = 2
    net.n[0] = eaData.X.shape[0]  # nx
    net.n[1] = 10
    net.n[2] = 3
    net.n[net.L] = 3

    net.activFun[1] = eaActivFun.relu
    net.activFun[2] = eaActivFun.softmax

    net.costFun = eaCostFun.L4
    return net

def demo3Classify():
    eaData = eaData3Classify()
    net = eaNetSoftmax(eaData)

    nnInitWb(net)
    nnFit(eaData, net, learn_rate=0.1)

    Ea.show(net, 3)

    # plotPredict(net, eaData.X, eaData.Type, cmap=['#0099CC', '#FF6666','#6622FF'])
    plotPredict(net, eaData.X, eaData.Type, cmap=['r', 'g', 'b'])
    plotCost("Logistics Costs", net.costs,netInfo=eaNetInfo(net))


if __name__ == '__main__':
    demo3Classify()