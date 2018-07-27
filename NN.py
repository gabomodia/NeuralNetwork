import random as rnd
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
        self.b_1 = [(rnd.random() - 0.5) for i in range(self.n_hidden)]
        self.b_2 = [(rnd.random() - 0.5) for i in range(self.n_outputs)]
        self.w_1 = np.array([[(rnd.random() - 0.5) for i in range(self.n_hidden)] for j in range(self.n_inputs)])
        self.w_2 = np.array([[(rnd.random() - 0.5) for i in range(self.n_outputs)] for j in range(self.n_hidden)])

    def train(self, X, T, eta, epochs=1):
        for i in range(epochs):
            print("Comenzando epoch %d", i)
            orden = list(range(len(X)))
            rnd.shuffle(orden)
            for j in range(len(X)):
                x = X[orden[j]]
                t = T[orden[j]]
                y = self.forward_propagate(x)

                # Comprobamos si está bien clasificada
                if np.argmax(y[1]) != np.argmax(t):

                    # Obtenemos los errores
                    delta_2 = np.array([y[1][k]-t[k] for k in range(len(y))])
                    delta_1 = np.matmul(np.matmul(np.diag(y[0]), self.w_2), delta_2)

                    # Descenso de gradiente para el sesgo
                    self.b_2 += eta * delta_2
                    self.b_1 += eta * delta_1

                    # Descenso de gradiente para los pesos
                    for k in range(len(self.w_2)):
                        self.w_2[k] += eta * delta_2 * y[0][k]
                    for k in range(len(self.w_1)):
                        self.w_1[k] += eta * delta_1 * x[k]

        print("Entrenamiento completado")
    def forward_propagate(self, x):
        y_1 = [self.activation((np.matmul(x, self.w_1)+self.b_1)[i]) for i in range(self.n_hidden)]
        y_2 = [self.activation((np.matmul(y_1, self.w_2)+self.b_2)[i]) for i in range(self.n_outputs)]
        return [y_1, y_2]

    def classify(self, x):
        return np.argmax(self.forward_propagate(x)[1])

