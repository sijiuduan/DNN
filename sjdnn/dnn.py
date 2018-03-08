from activeFunctions import *
from costFunctions import *
from ea import Ea
from testData import *
import matplotlib.pyplot as plt

class dnn():
    def __init__(self):
        pass

    def initWb(self, L, n):
        W = Ea()
        b = Ea()
        p = 0.2
        for l in range(1, L + 1):
            np.random.seed(2)
            W[l] = np.random.normal(size=(n[l], n[l - 1])) * p
            b[l] = np.zeros(shape=(n[l], 1))
        return W,b

    # DNN 网络 根据网络模型，预测结果
    def predict(self,L, W, b, g, X):
        A = Ea()
        A[0] = X
        Z = Ea()

        for l in range(1, L + 1):
            Z[l] = np.dot(W[l], A[l - 1]) + b[l]
            A[l] = g[l](Z[l])
        yhat = A[L]
        return yhat

    def run(self,X,Y,L,W,b,g,dg,J,dJ,learn_rate,max_itr=5000,tg=0.0001):
        # max_itr: 最大迭代次数
        # tg: 梯度截断参数
        output_cost = []

        A=Ea()
        A[0]=X
        dA = Ea()
        dZ = Ea()
        dW = Ea()
        db = Ea()
        Z=Ea()

        break_flag = 0
        old_cost = 0

        for i in range(1, max_itr):
            # forward properation
            for l in range(1, L + 1):
                Z[l] = np.dot(W[l], A[l - 1]) + b[l]
                A[l] = g[l](Z[l])

            cost = J(A[L], Y)
            # print(cost)
            dA[L] = dJ(A[L], Y)
            output_cost.append(cost)
            if old_cost - cost < tg * output_cost[0]:
                break_flag += 1

            old_cost = cost

            # back properation
            for l in range(0, L):
                l = L - l
                _dg = dg[l](Z[l])
                dZ[l] = _dg * dA[l]
                dA[l - 1] = np.dot(W[l].T, dZ[l])
                dW[l] = 1 / X.shape[1] * np.dot(dZ[l], A[l - 1].T)
                db[l] = 1 / X.shape[1] * np.sum(dZ[l], axis=1, keepdims=True)

            # update parameters
            for l in range(0, L):
                l = L - l
                W[l] = W[l] - learn_rate * dW[l]
                b[l] = b[l] - learn_rate * db[l]

            if break_flag > 30:
                break

        return output_cost

    def showCosts(self, costs,L,n,g,J):
        g_names = []
        for l in range(1, L + 1):
            g_names.append(g[l].__name__)

        plt.figure(figsize=(7, 7), dpi=56)

        plt.title("Hide Layers:%d \n" \
                  "Notes of Layers:[%s] \n" \
                  "J(A,Y):%s \n" \
                  "g(Z):[%s]" % (L,
                                 ", ".join(map(str, n[0:L + 1])),
                                 J.__name__,
                                 " -> ".join(g_names)), loc='left')

        plt.grid(True)
        wd = len(costs)
        plt.plot(range(0, len(costs)), costs, label='J (A,Y)')
        plt.ylabel("Cost", fontsize=14)
        plt.xlabel("Iterators", fontsize=14)
        plt.text(wd * .8, (costs[0] - costs[-1]) * .33, "Cost:%f" % costs[-1])
        plt.legend()
        plt.show()

    def tfDemo(self):
        tf = tf_data()
        # plot_tf(tf)
        X = tf[:-1]
        Y = tf[-1:]

        L = 2  # 或 L=3 试试
        n = [0, 20, 4, 1]  # 定义每层神经节点个数
        n[0] = X.shape[0]
        n[L] = 1  # 前置输出层一个节点

        g = Ea()  # 定义激活函数
        g[1] = relu
        g[2] = relu
        g[L] = sigmoid

        dg = Ea()  # 定义激活导函数
        dg[1] = dz_relu
        dg[2] = dz_relu
        dg[L] = dz_sigmoid

        J = L3  # 定义 代价函数 (试试 L2)
        dJ = dL3  # 定义 代价导函数 (试试 dL2)

        self.W, self.b = self.initWb(L, n)

        costs = self.run(X, Y, L, self.W, self.b, g, dg, J, dJ, 0.1, max_itr=100000, tg=0.00001)

        Yhat = self.predict(L, self.W, self.b, g, X)
        Yhat = np.array(Yhat > 0.5)
        Y = np.array(Y > 0.5)
        result = np.int32(Y ^ Yhat)

        print("判断错误的数据有%d条，占总数据的 %.2f%%" % (np.sum(result), 100 * np.sum(result) / X.shape[1]))
        print("DNN 迭代%d次，代价函数收敛于%.4f." % (len(costs),costs[-1]))

        x = X
        cm_light = ListedColormap(['#0099CC', '#FF6666'])

        def extend(a, b):
            return 1.05 * a - 0.05 * b, 1.05 * b - 0.05 * a

        x1_min, x1_max = extend(x[0].min(), x[0].max())  # x1的范围
        x2_min, x2_max = extend(x[1].min(), x[1].max())  # x2的范围

        print(x1_min, x1_max)
        print(x2_min, x2_max)

        N = 500; M = 500
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)

        print(x1.shape)

        x_show = np.stack((x1.flat, x2.flat), axis=1).T
        y_hat = self.predict(L, self.W, self.b, g, x_show)
        y_hat = np.int32(np.array(y_hat > 0.5))
        y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
        plt.figure(facecolor='w')
        print(x1.shape,x2.shape,y_hat.shape)
        plt.pcolormesh(x1, x2, y_hat, cmap=cm_light, alpha=0.4)

        Y = np.squeeze(Y)
        plt.scatter(X[0], X[1], s=30, c=Y, edgecolors='k', cmap=cm_light)



        # plt.scatter(x[0], x[1], s=30, c=y_hat, edgecolors='k', cmap=cm_light)  # 样本的显示

        plt.show()

        print(y_hat)




        # self.showCosts(costs,L,n,g,J)


    def lineDemo(self):
        X, Y = linear_data()
        L = 1
        n = [0,4,4,4,1]
        n[0] = X.shape[0]
        n[L] = 1

        g = Ea()
        dg = Ea()
        for l in range(1, L + 1):
            g[l] = linear
            dg[l] = dz_linear

        J = cfn.L2
        dJ = cfn.dL2

        self.W, self.b = self.initWb(L, n)

        costs = self.run(X, Y, L, self.W, self.b, g, dg, J, dJ, 0.02,tg=0.0000001)

        Ea.show(self.W)
        Ea.show(self.b)

        print(costs[-1])

        self.showCosts(costs)

if __name__ == '__main__':
    dnn = dnn()

    # dnn.lineDemo()
    dnn.tfDemo()
