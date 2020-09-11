import numpy as np

from learning_rule import LearningRule


class DeltaRule(LearningRule):
    def fit_batch(self, W, X, T, eta, n_epoch):
        for epoch in range(n_epoch):
            delta = - eta * (W.dot(X) - T).dot(X.T)
            error = np.linalg.norm(W.dot(X) - T, ord=2)
            W += delta
            print(">epoch=%s, learning rate=%s, error=%.2f" % (epoch, eta, error))

            self.error_array.append(error)
            if error <= 0.5:
                print("\nTraining finished in %s epoch\n" % epoch)
                break
            elif error > np.exp(10):
                raise RuntimeError("Error becomes nan")
        return W

    def fit_sequential(self, W, X, T, eta, n_epoch):
        # m features, n samples
        m, n = X.shape[0], X.shape[1]
        out_dim = 1 if len(T.shape) == 1 else T.shape[0]
        threshold = n * m * out_dim * 0.01
        for e in range(n_epoch):
            sum_error = 0.0
            for t in range(n):  # iterates over each sample
                for i in range(m):  # iterates over each feature
                    for j in range(out_dim):  # iterates over each output dimension
                        x = X.T[t]  # sample: one-by-m array
                        activation = W[j].dot(x)
                        error = T[j][t] - activation
                        delta = eta * x[i] * error
                        W[j][i] += delta
                        sum_error += 0.5 * error ** 2
            self.error_array.append(sum_error)
            print(">epoch=%s, learning rate=%s, error=%.2f" % (e, eta, sum_error))
            if sum_error < threshold:
                print("\nTraining finished in %s epoch\n" % e)
                break
        return W