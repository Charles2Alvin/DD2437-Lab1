import numpy as np


class LearningRule:
    debug = False

    def fit_batch(self, W, X, T, eta, n_epoch):
        pass

    def fit_sequential(self, W, X, T, eta, n_epoch):
        pass


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

            if sum_error == 0.0:
                print("\nTraining finished in %s epoch\n" % epoch)
                break


class DeltaRule(LearningRule):
    def fit_batch(self, W, X, T, eta, n_epoch):
        for epoch in range(n_epoch):
            delta = - eta * (W.dot(X) - T).dot(X.T)
            error = np.linalg.norm(W.dot(X) - T, ord=2)
            W += delta
            print(">epoch=%s, learning rate=%s, error=%.2f" % (epoch, eta, error))

            if error <= 0.5:
                print("\nTraining finished in %s epoch\n" % epoch)
                break
            elif error > np.exp(10):
                raise RuntimeError("Error becomes nan")
        return W

    def fit_sequential(self, W, X, T, eta, n_epoch):
        # m features, n samples
        m, n = X.shape[0], X.shape[1]
        output_dim = 1 if len(T.shape) == 1 else T.shape[0]
        threshold = n * m * output_dim * 0.01
        print("threshold, ", threshold)
        for e in range(n_epoch):
            sum_error = 0.0
            for t in range(n):  # iterates over each sample
                for i in range(m):  # iterates over each feature
                    for j in range(output_dim):  # iterates over each output dimension
                        x = X.T[t]  # sample: one-by-m array
                        activation = W[j].dot(x)
                        error = T[j][t] - activation
                        delta = eta * x[i] * error
                        W[j][i] += delta
                        sum_error += 0.5 * error ** 2
            print(">epoch=%s, learning rate=%s, error=%.2f" % (e, eta, sum_error))
            if sum_error < threshold:
                print("\nTraining finished in %s epoch\n" % n_epoch)
                break

        return W
