import time

from sklearn.metrics import accuracy_score

from perceptron import Perceptron
from prepare_data import DataProducer

X = DataProducer.produce_bits()
model = Perceptron(eta=0.1, algorithm='backprop')
start = time.time()
model.fit(X, X, n_epoch=1000, mode='batch', plot=False)
end = time.time()
pred = model.predict(X)
accuracy = accuracy_score(X, pred)
print("Prediction: %s" % pred)
print("Accuracy: %.3f" % accuracy)

