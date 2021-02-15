"""coding: utf-8"""
import numpy as np
from numpy import array
from math import sqrt


'''Média'''


def mean(v):
    if len(v.shape) == 1:
        return sum(v) / len(v)

    return array([sum(v.T[n]) / v.T[n].shape[0] for n in range(v.shape[1])])


'''Matriz de correlação de Pearson'''


def mat_corelation(m):
    m_cor = np.zeros(m.shape[1] ** 2).reshape(m.shape[1], m.shape[1])
    for e_x in range(m.shape[1]):
        for e_y in range(m.shape[1]):
            m_cor[e_x][e_y] = round(covariance(m.T[e_x], m.T[e_y]) / (standard_deviation(m.T[e_x]) *
                                                                      standard_deviation(m.T[e_y])), 1)
    return m_cor


'''Matriz de covariância'''


def mat_covarience(m):
    m_cov = np.zeros(m.shape[1] ** 2).reshape(m.shape[1], m.shape[1])
    for e_x in range(m.shape[1]):
        for e_y in range(m.shape[1]):
            m_cov[e_x][e_y] = covariance(m.T[e_x], m.T[e_y])
    return m_cov


'''Covariância entre dois atributos'''


def covariance(v_x, v_y):
    if v_x.shape[0] != v_y.shape[0]:
        return None
    cov = np.zeros(v_x.shape[0])

    mx = v_x.mean()
    my = v_y.mean()
    for el in range(v_x.shape[0]):
        cov[el] = (v_x[el] - mx) * (v_y[el] - my)
    return sum(cov) / v_x.shape[0]


'''Variância'''


def variance(v):
    v = v.copy()

    if len(v.shape) == 1:
        mx = v.mean()
        for n in range(v.shape[0]):
            v[n] = abs(v[n] - mx) ** 2
        return sum(v) / len(v)

    for c in range(v.shape[1]):
        mx = v.T[c].mean()
        for n in range(v.shape[0]):
            v.T[c][n] = (v.T[c][n] - mx) ** 2

    return array([sum(v.T[n]) / v.shape[0] for n in range(v.shape[1])])


'''Desvio padrão'''


def standard_deviation(m):
    if len(m.shape) == 1:
        return sqrt(mean(deviation(m)))

    return np.array([sqrt(mean(deviation(m.T[n]))) for n in range(m.shape[1])])


'''Desvio: Xi - mean(X)'''


def deviation(v):
    return np.array([((v[n] - mean(v)) ** 2) for n in range(v.shape[0])])


'''Normaliza os dados por Z Score'''


def z_score(dat):
    v = dat.copy()
    if len(dat.shape) == 1:
        return [((v[n] - mean(dat)) / np.std(dat)) for n in range(len(v))]

    for c in range(v.shape[1]):
        for n in range(v.shape[0]):
            v.T[c][n] = (dat.T[c][n] - mean(dat.T[c])) / np.std(dat.T[c])
    return v


'''Normaliza os dados (0, 1)'''


def normalize_min_max(dat):
    v_x = dat.copy()

    if len(v_x.shape) == 1:
        return [(v_x[n] - min(dat)) / (max(dat) - min(dat)) for n in range(len(dat))]

    for c in range(v_x.shape[1]):
        for n in range(v_x.shape[0]):
            v_x.T[c][n] = (dat.T[c][n] - min(dat.T[c])) / (max(dat.T[c]) - min(dat.T[c]))
    return v_x
