from ea import Ea
import activeFunctions as afn
import costFunctions as cfn
import numpy as np

# # Hyperparameters
# L = 4 # 神经网络层数
# learn_rate = 0.02 # 学习率 （有些书里，学习率用alpha表示）
# iterators = 5000  # 迭代次数
# n = Ea()
# # n[0] = X.shape[0] # 第0层是输入层。
# n[1] = 4 # 第1层 4 个节点
# n[2] = 4 # 第2层 4 个节点
# n[3] = 4 # 第3层 4 个节点
# n[4] = 1 # 第4层 1 个节点
# n[L] = 1 # 强制输出层 1 个节点
#
# g = Ea() # 分层定义激活函数
# g[1] = afn.linear
# g[2] = afn.linear
# g[3] = afn.linear
# g[4] = afn.linear
#
# dg = Ea() # 分层定义激活导函数
# dg[1] = afn.dz_linear
# dg[2] = afn.dz_linear
# dg[3] = afn.dz_linear
# dg[4] = afn.dz_linear
#
# loss = cfn.L2 # 定义成本函数
# dA_loss = cfn.dA_L2 # 定义成本导函数


def hyperParameters(L,learn_rate,nx,nl,gl,dgl,cfn,dA_cfn):
    hp = Ea()
    hp.L = L
    hp.learn_rate = learn_rate
    hp.n[0] = nx
    for l in range(1, L+1):
        hp.n[l] = nl[l-1]
        hp.g[l] = gl[l-1]
        hp.dg[l] = dgl[l - 1]
    hp.n[L] = 1

    hp.cost = cfn
    hp.dA_cost = dA_cfn

    return hp

def wbParameters(hp: Ea, r=0.2):
    pa = Ea()
    for l in range(1, hp.L + 1):
        np.random.seed(2)
        pa.W[l] = np.random.normal(size=(hp.n[l], hp.n[l - 1])) * r
        # print(np.sum(pa.W[l]))
        pa.b[l] = np.zeros(shape=(hp.n[l], 1))

    return pa




if __name__ == '__main__':
    hp = hyperParameters(L=4,learn_rate=0.02, iterators=5000, nx=2, nl=[4,4,4,1],
              gl=4*[afn.linear],
              dgl=4*[afn.dz_linear],
              cfn=cfn.L2,
              dA_cfn=cfn.dA_L2)
    Ea.show(hp)

    wb = wbParameters(hp,1)
    Ea.show(wb)
    pass

