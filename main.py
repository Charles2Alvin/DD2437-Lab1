import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from perceptron import Perceptron
from prepare_data import DataProducer

if __name__ == '__main__':
    n = 100
    patterns, targets = DataProducer.produce_binary(n, plot=False)
    X_train, X_test, y_train, y_test = train_test_split(patterns.T, targets, test_size=0.2, random_state=1)
    X_train, X_test = X_train.T, X_test.T
    model = Perceptron(eta=0.3, algorithm='backprop')
    start = time.time()
    model.fit(X_train, y_train, n_epoch=2000, mode='batch', plot=False)
    end = time.time()
    model.plot_learning_curve()
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("Weights: %s" % model.getWeights())
    print("Accuracy: %.3f" % accuracy)
    print("Time consumed: %.3fs" % (end - start))
