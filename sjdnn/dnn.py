import activeFunctions as af
import costFunctions as cf
from activeFunctions import *
from costFunctions import *
from dnnParameters import *
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

    def run(self,X,Y,L,W,b,g,dg,J,dJ,learn_rate,max_itr=5000,tg=0.0001):
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

            if break_flag > 3000:
                break

        return output_cost

    def showCosts(self, costs):
        plt.title("Cost with iterators ")
        plt.grid(True)
        wd = len(costs)
        plt.plot(range(1, wd - 1), costs[1:wd - 1], label='J (A,Y)')
        plt.ylabel("Cost", fontsize=14)
        plt.xlabel("Iterators", fontsize=14)
        plt.text(wd * .8, costs[0] * .1, "Cost:%f" % costs[-1])
        plt.legend()
        plt.show()

    def tfDemo(self):
        t,f = tf_data(1000)
        tf = np.c_[t, f]
        X = tf[:2]
        Y = tf[2:]

        # plotGR(t,f)

        L=2
        n=[0,20,2,10,10,10,10]
        n[0] = X.shape[0]
        n[L] = 1

        g = Ea()
        dg = Ea()

        g[1] = leaky_relu
        g[2] = leaky_relu
        g[3] = leaky_relu
        g[4] = leaky_relu
        g[5] = leaky_relu
        g[L] = sigmoid

        dg[1] = dz_leaky_relu
        dg[2] = dz_leaky_relu
        dg[3] = dz_leaky_relu
        dg[4] = dz_leaky_relu
        dg[5] = dz_leaky_relu
        dg[L] = dz_sigmoid


        J = L3
        dJ = dL3

        self.W, self.b = self.initWb(L, n)

        costs = self.run(X, Y, L, self.W, self.b, g, dg, J, dJ, 0.1, max_itr=100000, tg=0.001)

        print(costs[-1])
        self.showCosts(costs)


    def lineDemo(self):
        X, Y = linear_data()
        L = 1
        n = [0,4,4,4,1]
        n[0] = X.shape[0]
        n[L] = 1

        g = Ea()
        dg = Ea()
        for l in range(1, L + 1):
            g[l] = afn.linear
            dg[l] = afn.dz_linear

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
