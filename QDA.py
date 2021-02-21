"""coding: utf-8"""
import funcoes as f
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def z_score(dat):
    v = dat.copy()
    if len(dat.shape) == 1:
        return [((v[n] - mean(dat)) / np.std(dat)) for n in range(len(v))]

    for cl in range(v.shape[1]):
        for n in range(v.shape[0]):
            v.T[cl][n] = (dat.T[cl][n] - mean(dat.T[cl])) / np.std(dat.T[cl])
    return v


def mean(m):
    if len(m.shape) == 1:
        return np.mean(m)
    return np.array([[np.mean(v) for v in m]]).T


def qda(ts, atr_tr, cls_tr):
    ts = np.array([ts]).T
    cls = []
    for el in range(len(cls_tr)):
        if el == 0:
            cls.append(cls_tr.T[0][el])
        else:
            if sum([cls_tr[el] == cls[i] for i in range(len(cls))]) == 0:
                cls.append(cls_tr.T[0][el])

    cls_p = np.array([cls, np.zeros(len(cls))])

    df = pd.DataFrame(np.vstack((atr_tr.T, cls_tr.T)).T, columns=i_atr)

    for cl in range(cls_p.shape[1]):
        c_i = df[df.clas == cls_p[0][cl]].values.T[:df.shape[1] - 1]

        cov = np.cov(c_i)
        m_i = mean(c_i)

        cls_p[1][cl] = (np.log(np.linalg.det(cov) + 1e-50) +
                        ((ts - m_i).T @ np.linalg.pinv(cov) @ (ts - m_i)) - (2 * np.log(c_i.shape[1] / df.shape[0])))

    cl_tst = 0
    for i in range(cls_p.shape[1]):
        if cls_p[1][i] < cls_p[1][cl_tst]:
            cl_tst = i

    return cls_p[0][cl_tst]


if __name__ == '__main__':

    data_f = f.get_data("data_dermato_03.txt")

    i_atr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
             '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', 'clas']
    '''

    #   Naive Bayes Classifier 
    #   K-fold com 10 grupos
    #   Taxa de erro:  0.82%

    i_atr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
             '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', 'clas']
    '''

    '''
    #   Naive Bayes Classifier
    #   K - fold com 5 grupos
    #   Taxa de erro:  5.21%'''
    '''
    i_atr = ['1', '2', '3', '7', '9', '10', '14', '15', '16', '19', '20', '21', '22', '23', '24', '26', '28', '30',
             '31', '34', 'clas']


    '''
    #   i_atr = ['1', '2', '3', '4', '5', '11', '13', '14', '15', '17', '18', '32', 'clas']
    #   i_atr = ['2', '4', '5', '11', '13', '14', '15', '17', '18', '32', 'clas']

    data = []

    for c in i_atr:
        if c == 'clas':
            data.append(data_f.T[data_f.shape[1] - 1])
        else:
            data.append(data_f.T[int(c) - 1])
    data = np.array(data).T
    #   data = f.z_score(data)
    #   data = f.normalize_min_max(data)
    atributos = data.T[:data.shape[1] - 1].T

    classes = data.T[data.shape[1] - 1:].T

    atributos, classes = shuffle(atributos, classes, random_state=0)

    #  Quantidade de grupos no K-fold
    k = 5

    '''Dados para treino e teste'''
    atributos_teste = np.ones((atributos.shape[0] // k) * atributos.shape[1]).reshape(atributos.shape[0] // k,
                                                                                      atributos.shape[1])
    classes_teste = np.ones(atributos.shape[0] // k)

    atributos_treino = np.ones((atributos.shape[0] - (atributos.shape[0] // k)) * atributos.shape[1]).reshape(
        (atributos.shape[0] - (atributos.shape[0] // k)), atributos.shape[1])

    classes_treino = np.ones((atributos.shape[0] - (atributos.shape[0] // k)))

    k_f_results = np.zeros(k)

    '''K-fold com 5 grupos'''
    for k_f in range(k):

        '''Segmenta por indexação os dados de treino e teste'''
        atributos_teste = atributos[k_f * (atributos.shape[0] // k): (k_f + 1) * (atributos.shape[0] // k)]
        classes_teste = classes[k_f * (atributos.shape[0] // k): (k_f + 1) * (atributos.shape[0] // k)]

        if k_f == 0:
            atributos_treino = atributos[(k_f + 1) * (atributos.shape[0] // k):]
            classes_treino = classes[(k_f + 1) * (atributos.shape[0] // k):]
        elif (atributos.shape[0] % k) == 0 and k_f == k - 1:
            atributos_treino = atributos[:k_f * (atributos.shape[0] // k)]
            classes_treino = classes[:k_f * (atributos.shape[0] // k)]
        else:
            atributos_treino[:k_f * (atributos.shape[0] // k)] = atributos[:k_f * (atributos.shape[0] // k)]
            atributos_treino[k_f * (atributos.shape[0] // k):] = atributos[
                                                                 (k_f + 1) * (atributos.shape[0] // k):]

            classes_treino[:k_f * (atributos.shape[0] // k)] = classes[:k_f * (atributos.shape[0] // k)]
            classes_treino[k_f * (atributos.shape[0] // k):] = classes[(k_f + 1) * (atributos.shape[0] // k):]

        '''Classifica as amostras de teste, e armazena os resultados no vetor result'''
        result = [qda(atributos_teste[t], atributos_treino, classes_treino)
                  for t in range(atributos_teste.shape[0])]

        '''Verifica a taxa de erro e armazena em cada rodada do k-fold'''
        k_f_results[k_f] = sum([0 == i for i in [classes_teste[i] == result[i] for i in
                                                 range(len(classes_teste))]]) / len(classes_teste)

    '''Imprime a média das taxas de erro das rodadas do k-fold'''
    print(f'\nNaive Bayes Classifier \nK-fold com {k} grupos\nTaxa de erro: {f.mean(k_f_results) * 100: .2f}%')
