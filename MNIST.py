import NN
import random as rnd
import numpy as np
import os.path as path
from sklearn.datasets import fetch_mldata


def preparar_entrada():
    if not path.exists("entrenamiento.dat"):
        mnist = fetch_mldata("MNIST original", data_home="/home/gabriel/Documentos/Gecom")
        entrenamiento = []
        te = []
        test = []
        tt = []
        valores = rnd.sample(range(70000), 56000)
        for i in range(70000):
            c = np.array(mnist.data[i]).reshape(784)
            if i in valores:
                entrenamiento.append(c)
                te.append(mnist.target[i])
            else:
                test.append(c)
                tt.append(mnist.target[i])
        np.savetxt("entrenamiento.dat", entrenamiento)
        np.savetxt("te.dat", te)
        np.savetxt("test.dat", test)
        np.savetxt("tt.dat", tt)
    else:
        entrenamiento = np.loadtxt("entrenamiento.dat")
        te = np.loadtxt("te.dat")
        test = np.loadtxt("test.dat")
        tt = np.loadtxt("tt.dat")
    return entrenamiento, te, test, tt


nn = NN.MultilayerPerceptron(784, 397, 10)
entrenamiento, te, test, tt = preparar_entrada()
nn.train(entrenamiento, te, 0.1, 5)
prec = 0
for i in range(len(test)):
    estimado = nn.classify(test[i])
    if estimado == tt[i]:
        prec += 1

print("Precicion: %d", prec / len(test) * 100)


