import numpy as np

from learning_rule import LearningRule, phi, phi_d


class AutoEncoder(LearningRule):
    n_hidden_nodes: int
    dim = 0      # the output dimension
    alpha = 0.9  # for momentum

    def __init__(self, n_hidden_nodes):
        self.n_hidden_nodes = n_hidden_nodes
        self.n_layer = 2
        self.W = None
        self.V = None

    def fit_batch(self, W, X, T, eta, n_epoch):
        # M features, N samples
        M, N = X.shape[0], X.shape[1]
        dim = 1 if len(T.shape) == 1 else T.shape[0]

        W = np.random.normal(loc=0, scale=1, size=(self.n_hidden_nodes, M))
        V = np.random.normal(loc=0, scale=1, size=(dim, self.n_hidden_nodes + 1))

        print(W.shape)
        alpha = self.alpha
        d_W = np.zeros(W.shape)
        d_V = np.zeros(V.shape)
        threshold = 0.001
        for epoch in range(n_epoch):
            # forward pass
            H_in = W.dot(X)
            H_out = np.row_stack((phi(H_in), np.ones(N)))

            O_in = V.dot(H_out)
            O_out = phi(O_in)

            # backward pass
            delta_o = np.multiply(O_out - T, phi_d(O_in))
            delta_h = np.multiply(V.T.dot(delta_o)[1:self.n_hidden_nodes + 1, :], phi_d(H_in))

            # weight update
            d_W = alpha * d_W - (1 - alpha) * (delta_h.dot(X.T))
            d_V = alpha * d_V - (1 - alpha) * (delta_o.dot(H_out.T))

            W += eta * d_W
            V += eta * d_V

            sum_error = 0.5 * np.linalg.norm((T - O_out), ord=2)
            self.error_array.append(sum_error)

            print(">epoch=%s, learning rate=%s, error=%.2f" % (epoch, eta, sum_error))
            if sum_error < threshold:
                print("\nTraining finished in %s epoch\n" % epoch)
                break

        self.W = W
        self.V = V
        self.dim = dim

    def fit_sequential(self, W, X, T, eta, n_epoch):
        # M features, N samples
        M, N = X.shape[0], X.shape[1]
        dim = 1 if len(T.shape) == 1 else T.shape[0]

        W = np.random.normal(loc=0, scale=1, size=(self.n_hidden_nodes, M))
        V = np.random.normal(loc=0, scale=1, size=(dim, self.n_hidden_nodes + 1))
        H_in = np.zeros((self.n_hidden_nodes, 1))
        H_out = np.zeros((self.n_hidden_nodes + 1, 1))
        delta_h = np.zeros(H_out.shape)

        O_out = np.zeros((dim, 1))
        O_in = np.zeros(O_out.shape)
        delta_o = np.zeros(O_out.shape)

        d_W = 0.0
        d_V = 0.0
        alpha = self.alpha
        threshold = N * M * dim * 0.001
        for epoch in range(n_epoch):
            sum_error = 0.0
            for n in range(N):  # iterates over each sample
                # forward pass
                x = X.T[n]
                t = T[:, n]
                for k in range(self.n_hidden_nodes):
                    w = W[k]
                    H_in[k] = w.dot(x)
                    H_out[k] = phi(w.dot(x))
                H_out[-1][0] = 1
                for k in range(dim):
                    v = V[k]
                    O_in[k] = v.dot(H_out)
                    O_out[k] = phi(v.dot(H_out))
                    sum_error += 0.5 * (t[k] - O_out[k]) ** 2

                # backward pass
                for k in range(dim):
                    delta_o[k] = (O_out[k] - t[k]) * phi_d(O_in[k])

                for j in range(self.n_hidden_nodes):
                    delta_h[j] = (delta_o.dot(V[:, j])) * phi_d(H_in[j])

                # weight update
                for j in range(W.shape[0]):
                    for i in range(W.shape[1]):
                        d_W = alpha * d_W - (1 - alpha) * (x[i] * delta_h[j])
                        W[j][i] += eta * d_W

                for k in range(V.shape[0]):
                    for j in range(V.shape[1]):
                        d_V = alpha * d_V - (1 - alpha) * (H_out[j] * delta_o[k])
                        V[k][j] += eta * d_V

            self.error_array.append(sum_error)

            print(">epoch=%s, learning rate=%s, error=%.2f" % (epoch, eta, sum_error))
            if sum_error < threshold:
                print("\nTraining finished in %s epoch\n" % epoch)
                break

        self.W = W
        self.V = V
        self.dim = dim

    def predict(self, W: np.ndarray, X: np.ndarray):
        W, V = self.W, self.V
        X = np.row_stack((X, np.ones(X.shape[1])))
        H = phi(W.dot(X))
        H = np.row_stack((H, np.ones(H.shape[1])))
        Output = phi(V.dot(H))
        prediction = np.sign(Output)
        return prediction[0] if prediction.shape[0] == 1 else prediction
