import numpy as np
from matplotlib import pyplot as plt


class DataProducer:
    @classmethod
    def produce(cls, n: int = 100, plot: bool = False):
        """
        Produces two sets of points from multivariate normal distribution,
        and shuffle samples to get one data set

        :param n:
            The number of samples per class
        :param plot:
            True if need to plot the data-set, false otherwise
        :return:
            A data-set that contains linearly-separable data for binary classification
        """
        mean_a = [-5, -2]
        mean_b = [3, 6]
        cov_a = [[0.5, 0], [0, 0.5]]
        cov_b = cov_a
        data_a = np.random.multivariate_normal(mean_a, cov_a, n)
        data_b = np.random.multivariate_normal(mean_b, cov_b, n)
        data = np.append(data_a.T, data_b.T, axis=1)

        target_a = np.linspace(-1, -1, n)
        target_b = np.ones(n)
        target = np.append(target_a, target_b)

        # shuffle the samples
        permutation = np.random.permutation(data.shape[1])

        data = data[:, permutation]
        target = target[permutation]

        if plot:
            plt.scatter(data_a.T[0], data_a.T[1], edgecolors='b', marker='o')
            plt.scatter(data_b.T[0], data_b.T[1], edgecolors='r', marker='x')
            plt.show()

        return data, target
