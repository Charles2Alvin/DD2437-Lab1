import numpy as np

from learning_rule import LearningRule


class PerceptronRule(LearningRule):
    def fit_batch(self, W, X, T, eta, n_epoch):
        factor = 1 / (np.max(T) - np.min(T))

        for epoch in range(n_epoch):
            activation = W.dot(X)
            prediction = np.sign(activation)
            error = factor * (T - prediction)
            sum_error = np.linalg.norm(error, ord=1)
            delta = eta * error.dot(X.T)
            W += delta

            self.error_array.append(sum_error)
            print(">epoch=%s, learning rate=%s, error=%.2f" % (epoch, eta, sum_error))

            if sum_error == 0.0:
                print("\nTraining finished in %s epoch\n" % epoch)
                break

    def fit_sequential(self, W, X, T, eta, n_epoch):
        # n samples
        n = X.shape[1]

        # the output dimensions
        out_dim = 1 if len(T.shape) == 1 else T.shape[0]

        factor = 1 / (np.max(T) - np.min(T))

        for epoch in range(n_epoch):
            sum_error = 0.0
            for i in range(n):
                for j in range(out_dim):
                    x = X.T[i]
                    activation = W[j].dot(x.T)
                    prediction = np.sign(activation)
                    error = factor * (T[j][i] - prediction)
                    delta = eta * error * x

                    # update weights successively for each pattern
                    W[j] += delta
                    sum_error += abs(error)

                    if self.debug:
                        print("x1 = %0.3f,\t x2 = %0.3f,\t activation = %0.3f, \t predict=%s, "
                              "\t target=%s, " % (x[0], x[1], activation, prediction, T[j][i]))

            print(">epoch=%s, learning rate=%s, error=%.2f" % (epoch, eta, sum_error))

            self.error_array.append(sum_error)
            if sum_error == 0.0:
                print("\nTraining finished in %s epoch\n" % epoch)
                break