import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

""" this is the learning rate to be set by the user"""
alpha = 0.1

data = pd.read_csv("./train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
labels_dev = data_dev[0]
input_vals_dev = data_dev[1:n]
input_vals_dev = input_vals_dev / 255.0

data_train = data[1000:m].T
labels_train = data_train[0]
input_vals_train = data_train[1:n]
input_vals_train = input_vals_train / 255.0
_, m_train = input_vals_train.shape


def init_params():
    # 10 is the number of nodes in the layers of my network
    # 784 is the number of input nodes
    # 10 is the number of nodes in the next layer
    W1 = np.random.rand(10, 784) - 0.5
    """ 
        1 oh yeah because the function is something like 
        f(x) = w1.a1 + w2.a2 + w3.a3 + ... + wNaN + B <= this is one basis for each output node 
    """
    B1 = np.random.rand(10, 1) - 0.5
    # 10 , 10 because it goes from 10 nodes to 10 nodes
    W2 = np.random.rand(10, 10) - 0.5
    # same reason as above
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2


def ReLU(Z):
    # this will manually iterate through Z and do max(0, I) for each I in Z
    return np.maximum(0, Z)


def derivative_of_ReLU(Z):
    return Z > 0


def softmax(Z):
    """
    the bottom will give the constant number i.e. the sum of e^i for each i in Z
    then the top will put e^i / const in the place of i for each i in Z
    """
    # exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    # return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    # return np.exp(Z) / sum(np.exp(Z))
    # return Z / np.sum(Z, axis=0)
    A = np.exp(Z)
    return A / sum(A)


def forward_propogation(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def convert_to_2D(Y):
    """
    this will convert the 1D matrix to a 2D matrix like it will conver [0, 1, 2, 3, 9] to
    [ [1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,1] ]
    """
    # 10 because we want the size of each inner row to be 10 (0 - 9)
    Y2 = np.zeros((Y.size, 10))
    Y2[np.arange(Y.size), Y] = 1
    #  can also be done by
    # for i in range(Y.size):
    #     Y2[i][Y[i]] = 1
    Y2 = Y2.T
    return Y2


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    # m = Y.size
    Y_useful = convert_to_2D(Y)
    dZ2 = A2 - Y_useful
    dW2 = (1 / m) * dZ2.dot(A1.T)
    dB2 = (1 / m) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derivative_of_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    dB1 = (1 / m) * np.sum(dZ1)
    return dW1, dB1, dW2, dB2


def update_weights(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    return W1, B1, W2, B2


def get_predictions(A):
    return np.argmax(A, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    W1, B1, W2, B2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propogation(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, B1, W2, B2 = update_weights(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)
        if (i % 10) == 0:
            print("iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, B1, W2, B2


W1, B1, W2, B2 = gradient_descent(input_vals_train, labels_train, 1000, 0.5)
