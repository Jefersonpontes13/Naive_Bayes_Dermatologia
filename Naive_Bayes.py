"""coding: utf-8"""
import numpy as np
from numpy import array
import pandas as pd


def mean(x):
    if len(x.shape) == 1:
        return sum(x) / len(x)

    return [sum(x.T[n]) / x.T[n].shape[0] for n in range(x.shape[1])]


def mat_corelation(m):
    m_cor = np.zeros(m.shape[1] ** 2).reshape(m.shape[1], m.shape[1])
    for x in range(m.shape[1]):
        for y in range(m.shape[1]):
            m_cor[x][y] = correlation(m.T[x], m.T[y])
    return m_cor


def correlation(x, y):
    if x.shape[0] != y.shape[0]:
        return None
    cor = np.zeros(x.shape[0])

    for el in range(x.shape[0]):
        cor[el] = x[el] * y[el]
    return sum(cor) / x.shape[0]


def mat_covarience(m):
    m_cov = np.zeros(m.shape[1] ** 2).reshape(m.shape[1], m.shape[1])
    for x in range(m.shape[1]):
        for y in range(m.shape[1]):
            m_cov[x][y] = covariance(m.T[x], m.T[y])
    return m_cov


def covariance(x, y):
    if x.shape[0] != y.shape[0]:
        return None
    cov = np.zeros(x.shape[0])

    mx = x.mean()
    my = y.mean()
    for el in range(x.shape[0]):
        cov[el] = (x[el] - mx) * (y[el] - my)
    return sum(cov) / x.shape[0]


def variance(x):
    x = x.copy()

    if len(x.shape) == 1:
        mx = x.mean()
        for n in range(x.shape[0]):
            x[n] = abs(x[n] - mx) ** 2
        return sum(x) / len(x)

    for c in range(x.shape[1]):
        mx = x[c].mean()
        for n in range(x.shape[0]):
            x[c][n] = (x[c][n] - mx) ** 2

    return [sum(x[n]) / x.shape[0] for n in range(x.shape[1])]


if __name__ == '__main__':
    file = open("data_dermato_03.txt")
    data = file.readlines()
    file.close()

    for i in range(len(data)):
        data[i] = data[i].split()
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])

    data = array(data, dtype=np.float64)

    data_Frame = pd.DataFrame(
        {"erythema": data.T[0],
         "scaling": data.T[1],
         "definite_borders": data.T[2],
         "itching": data.T[3],
         "koebner_phenomenon": data.T[4],
         "polygonal_papules": data.T[5],
         "follicular_papules": data.T[6],
         "oral_mucosal_involvement": data.T[7],
         "knee_and_elbow_involvement": data.T[8],
         "scalp_involvement": data.T[9],
         "family_history": data.T[10],
         "Age": data.T[11],
         "melanin": data.T[11],
         "incontinence": data.T[12],
         "eosinophils_in_the_infiltrate": data.T[13],
         "PNL_infiltrate": data.T[14],
         "fibrosis_of_the_papillary_dermis,": data.T[15],
         "exocytosis": data.T[16],
         "acanthosis": data.T[17],
         "hyperkeratosis": data.T[18],
         "parakeratosis": data.T[19],
         "clubbing_of_the_rete_ridges": data.T[20],
         "elongation_of_the_rete_ridges": data.T[21],
         "thinning_of_the_suprapapillary_epidermis": data.T[22],
         "spongiform_pustule": data.T[23],
         "munro_microabcess": data.T[24],
         "focal_hypergranulosis": data.T[25],
         "disappearance_of_the_granular_layer": data.T[26],
         "vacuolisation_and_damage_of_basal_layer": data.T[27],
         "spongiosis": data.T[28],
         "saw_tooth_appearance_of_retes": data.T[29],
         "follicular_horn_plug": data.T[30],
         "perifollicular_parakeratosis": data.T[31],
         "inflammatory_monoluclear_inflitrate": data.T[32],
         "band_like_infiltrate": data.T[33],
         "class": data.T[34]
         }
    )
    atributos = data.T[:34]
    classes = data.T[34:]

    vet_media_atr = mean(atributos)
    vet_variancia_atr = variance(atributos)
    mat_covariancia = mat_covarience(atributos)
    mat_correlacao = mat_corelation(atributos)
