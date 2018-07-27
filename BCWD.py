from sklearn.datasets import load_breast_cancer
import numpy as np
import random as rnd
import NN


def preparar_entrada(bdatos):
    entrenamiento = []
    te = []
    test = []
    tt = []
    valores = rnd.sample(range(len(bdatos.data)), int(len(bdatos.data) * 0.8))
    for i in range(len(bdatos.data)):
        c = np.array(bdatos.data[i])
        c = c.reshape(30)
        if i in valores:
            entrenamiento.append(c)
            te.append(bdatos.target[i])
        else:
            test.append(c)
            tt.append(bdatos.target[i])
    return entrenamiento, te, test, tt


def transformartmnist(t):
    tt = np.ones(len(t))
    for i in range(len(t)):
        if t[i] == 0:
            tt[i] = -1
    return tt


def comparador(treal, tobtenido):
    resultadoscorrectos = 0
    resultadosincorrectos = 0
    for i in range(len(treal)):
        if treal[i] == tobtenido[i]:
            resultadoscorrectos += 1
        else:
            resultadosincorrectos += 1
    print("Aciertos: " + str(resultadoscorrectos) + ", Fallos: " + str(resultadosincorrectos) + " Porcentaje: ")
    print(resultadoscorrectos * 100 / len(treal))
    print("%")


bcwd = load_breast_cancer()
de, te, dt, tt = preparar_entrada(bcwd)
datosentrada = np.array(de)
datostest = np.array(dt)
tentrada = transformartmnist(te)
ttest = transformartmnist(tt)
nn = NN.MultilayerPerceptron(30, 15, 1)
nn.train(datosentrada, tentrada, 0.1, 5)
tobtenida = np.zeros(datostest.shape[0])
for i in range(datostest.shape[0]):
    tobtenida[i] = (1 if nn.classify(datostest[i]) >= 0 else -1)
comparador(ttest, tobtenida)
