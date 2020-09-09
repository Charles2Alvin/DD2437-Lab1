from sklearn.metrics import accuracy_score

from perceptron import Perceptron
from prepare_data import DataProducer

if __name__ == '__main__':
    n = 200
    patterns, targets = DataProducer.produce(n, plot=False)
    model = Perceptron(eta=0.01, algorithm='backprop')
    model.fit(patterns, targets, n_epoch=1000, mode='sequential', plot=False)
    pred = model.predict(patterns)
    accuracy = accuracy_score(targets, pred)
    print("Weights: %s" % model.getWeights())
    print("Accuracy: %.3f" % accuracy)
