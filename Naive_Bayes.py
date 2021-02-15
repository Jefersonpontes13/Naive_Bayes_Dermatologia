"""coding: utf-8"""
import funcoes as f
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def classifier_naive_bayes(ts, atr_tr, cls_tr):
    return None


if __name__ == '__main__':
    file = open("data_dermato_03.txt")
    data = file.readlines()
    file.close()

    for i in range(len(data)):
        data[i] = data[i].split()
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])

    data = array(data, dtype=np.float64)

    '''
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
    '''

    atributos = data.T[:34].T
    atributos = f.z_score(atributos)    # Normaliza os dados
    classes = data.T[34:].T

    atributos, classes = shuffle(atributos, classes, random_state=0)

    vet_media_atr = f.mean(atributos)
    vet_variancia_atr = f.variance(atributos)
    mat_covariancia = f.mat_covarience(atributos)
    mat_correlacao = f.mat_corelation(atributos)

    x = np.arange(mat_correlacao.shape[0])
    y = np.arange(mat_correlacao.shape[0])

    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, mat_correlacao)
    fig.show()

    k_k_f = 5
    k_f_results = np.zeros(5)

    '''Dados para treino e teste'''
    atributos_teste = np.ones((atributos.shape[0]//k_k_f) * atributos.shape[1]).reshape(atributos.shape[0]//k_k_f,
                                                                                        atributos.shape[1])
    classes_teste = np.ones(atributos.shape[0]//k_k_f)

    atributos_treino = np.ones((atributos.shape[0]//k_k_f) * (k_k_f - 1) * atributos.shape[1]).reshape(
        (atributos.shape[0]//k_k_f) * (k_k_f - 1), atributos.shape[1])
    classes_treino = np.ones((atributos.shape[0]//k_k_f) * (k_k_f - 1))

    '''K-fold com 5 grupos'''
    for k_f in range(5):

        '''Segmenta por indexação os dados de treino e teste'''
        atributos_teste = atributos[k_f * (atributos.shape[0]//k_k_f): (k_f + 1) * (atributos.shape[0]//k_k_f)]
        classes_teste = classes[k_f * (atributos.shape[0]//k_k_f): (k_f + 1) * (atributos.shape[0]//k_k_f)]

        if k_f == 0:
            atributos_treino = atributos[(k_f + 1) * (atributos.shape[0]//k_k_f):]
            classes_treino = classes[(k_f + 1) * (atributos.shape[0]//k_k_f):]
        elif k_f == k_k_f - 1:
            atributos_treino = atributos[:k_f * (atributos.shape[0]//k_k_f)]
            classes_treino = classes[:k_f * (atributos.shape[0]//k_k_f)]
        else:
            atributos_treino[:k_f * (atributos.shape[0]//k_k_f)] = atributos[:k_f * (atributos.shape[0]//k_k_f)]
            atributos_treino[k_f * (atributos.shape[0]//k_k_f):] = atributos[(k_f + 1) * (atributos.shape[0]//k_k_f):]
            classes_treino[:k_f * (atributos.shape[0]//k_k_f)] = classes[:k_f * (atributos.shape[0]//k_k_f)]
            classes_treino[k_f * (atributos.shape[0]//k_k_f):] = classes[(k_f + 1) * (atributos.shape[0]//k_k_f):]

        '''Classifica as amostras de teste, e armazena os resultados no vetor result'''
        result = [classifier_naive_bayes(test, atributos_treino, classes_treino) for test in atributos_teste]

        '''Verifica a taxa de erro e armazena em cada rodada do k-fold'''
        k_f_results[k_f] = sum([0 == i for i in classes_teste == result]) / len(classes_teste)

    '''Imprime a média das taxas de erro das rodadas do k-fold'''
    print('\n NPC \nK-fold com 10 grupos\n' + 'Taxa de erro: ' + str(f.mean(k_f_results)))
    exit()
