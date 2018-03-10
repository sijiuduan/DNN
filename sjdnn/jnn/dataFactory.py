from jnn.costFun import *
from jnn.ea import Ea
import math

def eaData3Classify(m=1000):
    distance = 8
    np.random.seed(1)
    X1 = 2 * np.random.randn(2, m)
    Y1 = np.zeros((3,m))
    Y1[0]=1
    type1 = np.zeros((1,m))

    X2 = 2 * np.random.randn(2, m)
    X2[0] += distance
    Y2 = np.zeros((3, m))
    Y2[1] = 1
    type2 = 1+np.zeros((1, m))

    X3 = 2 * np.random.randn(2, m)
    X3[0] += distance/2
    X3[1] += distance * math.sin(math.pi/3)
    Y3 = np.zeros((3, m))
    Y3[2] = 1
    type3 = 2 + np.zeros((1, m))

    X = np.c_[X1,X2,X3]
    Y = np.c_[Y1,Y2,Y3]
    Type = np.c_[type1,type2,type3]

    eaData = Ea()
    eaData.X = X
    eaData.Y = Y
    eaData.Type = Type

    return eaData

def eaDataTFRing(m=1000):
#   np.random.seed(...)  保证每次输出结果不变。
    np.random.seed(1)
    X = 2*np.random.randn(2,m)
#   很方便的求 2-范数 的方法（相当于求半径）
    R = np.linalg.norm(X,ord=2,axis=0).reshape(1,m)
#   增加一个噪音
    np.random.seed(2)
    noise = .3 * np.random.normal(size=R.shape)
    R = R + noise
#   二值化输出
    Y = np.int32(np.array(R < 2.5))
    print("dataTF  X.shape",X.shape,"Y.shape",Y.shape)
#   合并输入输出数据 （学会用 np.r_[...] 和 np.c_[...])
    eaData = Ea()
    eaData.X = X
    eaData.Y = Y
    eaData.Type = Y

    return eaData

def eaDataLinear(w,b,m=1000):
    nx = len(w)
    np.random.seed(1)
    X = np.random.normal(size=(nx, m))
    Y = np.dot(w, X) + b
    np.random.seed(2)
    noise = .01 * np.random.normal(size=Y.shape)
    # print(L2(noise,0))
    Y += noise

    eaData = Ea()
    eaData.X = X
    eaData.Y = Y

    return eaData