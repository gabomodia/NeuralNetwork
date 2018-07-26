from random import random
from math import exp
import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


class MultilayerPerceptron:
    def __init__(self, n_inputs, n_hidden, n_outputs, activation=sigmoid):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.activation = activation
        self.b_1 = [(random() - 0.5) for i in range(self.n_hidden)]
        self.b_2 = [(random() - 0.5) for i in range(self.n_outputs)]
        self.w_1 = np.array([[(random() - 0.5) for i in range(self.n_hidden)] for j in range(self.n_inputs)])
        self.w_2 = np.array([[(random() - 0.5) for i in range(self.n_outputs)] for j in range(self.n_hidden)])

    def train(self, X, T, eta, epochs=1):
        for i in range(epochs):
            for j in range(len(X)):
                y = self.forward_propagate(X[j])
                delta_2 = np.array([y[k]-T[j] for k in range(len(y))])
                delta_1 = np.diag(y)

    def forward_propagate(self, x):
        y_1 = [self.activation((np.matmul(x, self.w_1)+self.b_1)[i]) for i in range(self.n_hidden)]
        y_2 = [self.activation((np.matmul(y_1, self.w_2)+self.b_2)[i]) for i in range(self.n_outputs)]
        return [y_1, y_2]

        # y_1 = np.matmul([self.activation((np.matmul(x, self.w_1)+self.b_1)[i]) for i in range(self.n_hidden)], self.w_2)
        # return [self.activation((y_1+self.b_2)[i]) for i in range(self.n_outputs)]

    def classify(self, x):
        return np.argmax(self.forward_propagate(x)[1])


nn = MultilayerPerceptron(2, 3, 2)
resul = nn.classify(np.array([2, 2]))
print(resul)
