import numpy as np

def linear(z):
    return z
def dz_linear(z):
    return 1

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def dz_sigmoid(z):
    s = sigmoid(z)
    dz = s * (1-s)
    return dz

def tanh(z):
    return np.tanh(z)

def dz_tanh(z):
    t = tanh(z)
    dz = 1-t**2
    return dz

def relu(z):
    g = np.fmax(z, 0)
    return g

def dz_relu(z):
    g = np.fmax(z,0)
    g = np.sign(g)
    return g

def leaky_relu(z):
    g = np.fmax(z, 0.01*z)
    return g

def dz_leaky_relu(z):
    g = np.piecewise(z, [z < 0, z > 0], [.01, 1])
    return g

def softmax(z):
    x_exp = np.exp(z)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s

def softmax_derivative(x):
    assert(0)
    return x

def test():
    tt = np.array([
        [1.,1.,1.,1.], # hidden layer 1
        [1.,-1.,0.,0.], # hidden layer 2
        [-4.,5.,-1.,0.], # hidden layer 3
        [1.,0.,0.,0.]  # output layer 4
    ])

    g = dz_leaky_relu(tt)
    print(g)

if __name__ == '__main__':
    test()