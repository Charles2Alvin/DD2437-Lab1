import numpy as np
from matplotlib import pyplot as plt


class SingleLayerPerceptron:
    def __init__(self, eta: float = 0.01, algorithm: str = 'delta', debug: bool = False):
        # learning rate
        self.eta = eta

        # learning algorithm
        if algorithm not in ['perceptron', 'delta']:
            raise RuntimeError("No such algorithm: %s" % algorithm)
        self.algorithm = algorithm

        # launch mode: run or debug
        self.debug = debug

        # learned weight vector
        self.W = None

        # number of output dimensions
        self.out_dim = None

    @staticmethod
    def init_weight(n, m):
        return np.random.normal(loc=0, scale=1, size=(n, m))

    def fit(self, patterns: np.ndarray, targets: np.ndarray, n_epoch: int = 20, mode: str = 'sequential', plot: bool = False):
        if self.algorithm == 'delta':
            self.fit_delta_rule(patterns, targets, n_epoch)

        elif self.algorithm == 'perceptron':
            self.fit_perceptron_learning(patterns, targets, n_epoch, mode)

        if plot:
            self.plot_result(patterns, targets, self.W)

    def fit_perceptron_learning(self, X, T, n_epoch, mode):
        # m features, n samples
        m, n = X.shape[0], X.shape[1]

        # the output dimensions
        out_dim = 1 if len(T.shape) == 1 else T.shape[0]

        # add ones for threshold
        W = self.init_weight(1, m + 1)
        X = np.row_stack((X, np.ones(n)))
        T = T.reshape(out_dim, n)
        factor = 1 / (np.max(T) - np.min(T))

        if mode == 'sequential':
            for epoch in range(n_epoch):
                sum_error = 0.0
                for i in range(n):
                    for j in range(out_dim):
                        x = X.T[i]
                        activation = W[j].dot(x.T)
                        prediction = np.sign(activation)
                        error = factor * (T[j][i] - prediction)
                        delta = self.eta * error * x

                        # update weights successively for each pattern
                        W[j] += delta
                        sum_error += abs(error)

                        if self.debug:
                            print("x1 = %0.3f,\t x2 = %0.3f,\t activation = %0.3f, \t predict=%s, \t target=%s, "
                                  "\t delta=%s" % (x[0], x[1], activation, prediction, T[j][i], delta))

                print(">epoch=%s, learning rate=%s, error=%.2f, delta=%s" % (epoch, self.eta, sum_error, delta))

                if sum_error == 0.0:
                    print("\nTraining finished in %s epoch\n" % epoch)
                    break

        elif mode == 'batch':
            for epoch in range(n_epoch):
                activation = W.dot(X)
                prediction = np.sign(activation)
                error = factor * (T - prediction)
                sum_error = np.linalg.norm(error, ord=1)
                delta = self.eta * error.dot(X.T)
                W += delta

                print(">epoch=%s, learning rate=%s, error=%.2f" % (epoch, self.eta, sum_error))

                if sum_error == 0.0:
                    print("\nTraining finished in %s epoch\n" % epoch)
                    break
        else:
            raise RuntimeError("No such running mode")

        self.W = W
        self.out_dim = out_dim

    def fit_delta_rule(self, X, T, n_epoch):
        # m features, n samples
        m, n = X.shape[0], X.shape[1]

        output_dim = 1 if len(T.shape) == 1 else T.shape[0]

        W = self.init_weight(output_dim, m + 1)

        X = np.row_stack((X, np.ones(n)))

        for epoch in range(n_epoch):
            delta = - self.eta * (W.dot(X) - T).dot(X.T)
            error = np.linalg.norm(W.dot(X) - T, ord=2)
            W += delta
            print(">epoch=%s, learning rate=%s, error=%.2f" % (epoch, self.eta, error))

            if error <= 0.5:
                print("\nTraining finished in %s epoch\n" % epoch)
                break
            elif error > np.exp(10):
                raise RuntimeError("Error becomes nan")
        self.W = W
        self.out_dim = output_dim

    @staticmethod
    def plot_result(patterns, targets, w):
        n = patterns.shape[1]

        class_0 = np.array([patterns.T[i] for i in range(n) if targets[i] == -1]).T
        class_1 = np.array([patterns.T[i] for i in range(n) if targets[i] == 1]).T
        plt.title("Single layer perceptron learning")

        # plot the two classes
        plt.scatter(class_0[0], class_0[1], edgecolors='b', marker='o')
        plt.scatter(class_1[0], class_1[1], edgecolors='r', marker='x')

        # plot the decision boundary
        x = np.linspace(min(patterns[0]), max(patterns[0]))
        y = - (w[0][0] * x + w[0][2]) / w[0][1]
        plt.plot(x, y)
        plt.show()

    def predict(self, x_test: np.ndarray):
        n = x_test.shape[1]
        x_test = np.row_stack((x_test, np.ones(n)))
        prediction = np.sign(self.W.dot(x_test))
        return prediction[0] if self.out_dim == 1 else prediction

    def getWeights(self):
        return self.W[0] if self.out_dim == 1 else self.W
