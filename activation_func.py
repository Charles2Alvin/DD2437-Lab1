import numpy as np


def func1(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


def func1_d(x):
    return (1 - func1(x) ** 2) / 2


def relu(x):

    return np.maximum(0.0, x)


def relu_d(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

