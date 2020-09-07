import numpy as np
from matplotlib import pyplot as plt

from learning_rule import DeltaRule, PerceptronRule


class SingleLayerPerceptron:
    rule_map = {'perceptron': PerceptronRule(), 'delta': DeltaRule()}

    def __init__(self, eta: float = 0.01, algorithm: str = 'delta', debug: bool = False):
        # learning rate
        self.eta = eta

        # learning algorithm
        if algorithm not in self.rule_map:
            raise RuntimeError("No such algorithm: %s" % algorithm)
        self.rule = self.rule_map.get(algorithm)

        # launch mode: run or debug
        self.debug = debug

        # learned weight matrix: num(output dimensions) by num(features)
        self.W = None

        # number of output dimensions
        self.out_dim = None

    @staticmethod
    def init_weight(n, m):
        """
        Create an initial weight matrix filled with numbers drawn from normal distribution
        :param n:
        :param m:
        :return:
        """
        return np.random.normal(loc=0, scale=1, size=(n, m))

    def fit(self, X: np.ndarray, T: np.ndarray, n_epoch: int,
            mode: str = 'sequential', plot: bool = False):
        """
        Delegate the training task to the right function
        :param X: input patterns
        :param T: learning targets
        :param n_epoch: number of max epochs
        :param mode: learn samples sequentially or in batches
        :param plot: True if need to plot the data-set, false otherwise
        :return: None
        """
        # self.epoch = n_epoch

        # the output dimensions
        self.out_dim = 1 if len(T.shape) == 1 else T.shape[0]

        # m features, n samples
        m, n = X.shape[0], X.shape[1]

        # add ones as the bias trick
        W = self.init_weight(1, m + 1)
        X = np.row_stack((X, np.ones(n)))
        T = T.reshape(self.out_dim, n)

        if mode == 'batch':
            self.rule.fit_batch(W, X, T, self.eta, n_epoch)

        elif mode == 'sequential':
            self.rule.fit_sequential(W, X, T, self.eta, n_epoch)

        self.W = W
        if plot:
            self.plot_result(X, T, self.W)

    def predict(self, x_test: np.ndarray):
        n = x_test.shape[1]
        x_test = np.row_stack((x_test, np.ones(n)))
        prediction = np.sign(self.W.dot(x_test))
        return prediction[0] if self.out_dim == 1 else prediction

    def getWeights(self):
        return self.W[0] if self.out_dim == 1 else self.W

    @staticmethod
    def plot_result(patterns, targets, w):
        n = patterns.shape[1]

        class_0 = np.array([patterns.T[i] for i in range(n) if targets[0][i] == -1]).T
        class_1 = np.array([patterns.T[i] for i in range(n) if targets[0][i] == 1]).T
        plt.title("Single layer perceptron learning")

        # plot the two classes
        plt.scatter(class_0[0], class_0[1], edgecolors='b', marker='o')
        plt.scatter(class_1[0], class_1[1], edgecolors='r', marker='x')

        # plot the decision boundary
        x = np.linspace(min(patterns[0]), max(patterns[0]))
        y = - (w[0][0] * x + w[0][2]) / w[0][1]
        plt.plot(x, y)
        plt.show()

