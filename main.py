from prepare_data import DataProducer
from single_layer_perceptron import SingleLayerPerceptron
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    n = 300
    patterns, targets = DataProducer.produce(n, plot=False)
    model = SingleLayerPerceptron(eta=0.0001, algorithm='delta')
    model.fit(patterns, targets, n_epoch=1000, mode='sequential', plot=True)
    pred = model.predict(patterns)
    accuracy = accuracy_score(targets, pred)
    print("Weights: %s" % model.getWeights())
    print("Accuracy: %.3f" % accuracy)
