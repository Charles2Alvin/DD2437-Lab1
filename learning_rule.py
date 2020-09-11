import numpy as np
from matplotlib import pyplot as plt


def phi(x: str):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


def phi_d(x):
    return (1 - phi(x) ** 2) / 2


class LearningRule:
    error_array = []
    debug = False   # debug option

    def fit_batch(self, W, X, T, eta, n_epoch):
        pass

    def fit_sequential(self, W, X, T, eta, n_epoch):
        pass

    def predict(self, W: np.ndarray, x_test: np.ndarray):
        n = x_test.shape[1]
        x_test = np.row_stack((x_test, np.ones(n)))
        prediction = np.sign(W.dot(x_test))
        return prediction[0] if len(prediction.shape) == 1 else prediction

    def plot_learning_curve(self):
        x = np.linspace(1, len(self.error_array) + 1, num=len(self.error_array))
        plt.title("learning curve")
        plt.xlabel("epoch")
        plt.ylabel("Mean-squared error")
        plt.plot(x, self.error_array)
        plt.show()


