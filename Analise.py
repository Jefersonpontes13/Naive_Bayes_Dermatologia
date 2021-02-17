"""coding: utf-8"""
import funcoes as f
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = f.get_data("data_dermato_03.txt")

    atributos = data.T[:34].T
    classes = data.T[34:].T

    atributos, classes = shuffle(atributos, classes, random_state=0)

    vet_media_atr = f.mean(atributos)
    vet_variancia_atr = f.variance(atributos)
    mat_covariancia = f.mat_covarience(atributos)
    mat_correlacao = f.mat_corelation(atributos)

    x = np.arange(mat_correlacao.shape[0])
    y = np.arange(mat_correlacao.shape[0])
    for i in range(len(x)):
        x[i] = str(x[i] + 1)
        y[i] = str(y[i] + 1)

    fig, ax = plt.subplots()

    im = ax.imshow(
        [[abs(mat_correlacao[li][c]) for c in range(mat_correlacao.shape[1])] for li in range(mat_correlacao.shape[0])])

    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))

    ax.set_xticklabels(x)
    ax.set_yticklabels(y)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_title("Matriz de Correlação")
    fig.tight_layout()
    plt.savefig('correlacao.pdf')
    plt.show()
