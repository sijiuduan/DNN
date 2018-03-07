import numpy as np

# Cost Functions
def L1(A, Y):
    loss = np.sum(np.abs(A-Y))
    loss = np.squeeze(loss)
    return loss

def dL1(A,Y):
    return np.sign(A-Y)

def L2(A, Y):
    loss = 1/2 * np.dot((A-Y), (A-Y).T)
    loss = np.squeeze(loss)
    return loss

def dL2(A,Y):
    return A-Y

def L3(A, Y):
    m = Y.shape[1]
    loss = -1 / m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A), axis=1, keepdims=True)
    loss = np.squeeze(loss)
    assert (loss.shape == ())
    return loss

def dL3(A,Y):
    loss = -Y/A + (1-Y)/(1-A)
    loss = np.squeeze(loss)
    return loss

