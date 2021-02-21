"""coding: utf-8"""

import funcoes as f
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


from numpy.linalg import eig
from numpy.linalg import pinv
from numpy.linalg import inv


def get_data(file_name):
    file = open(file_name)
    dt = file.readlines()
    file.close()

    for line in range(len(dt)):
        dt[line] = dt[line].split()
        for el in range(len(dt[line])):
            dt[line][el] = int(dt[line][el])

    return np.array(dt, dtype=np.float64)


def z_score(dat):
    v = dat.copy()
    if len(dat.shape) == 1:
        return [((v[n] - mean(dat)) / np.std(dat)) for n in range(len(v))]

    for c in range(v.shape[1]):
        for n in range(v.shape[0]):
            v.T[c][n] = (dat.T[c][n] - mean(dat.T[c])) / np.std(dat.T[c])
    return v


'''Matriz de Espalhamento entre atributos'''


def mat_scarter(m):
    m_s = np.zeros(m.shape[0] ** 2).reshape(m.shape[0], m.shape[0])
    for e_x in range(m.shape[0]):
        for e_y in range(m.shape[0]):
            m_s[e_x][e_y] = scarter(m[e_x], m[e_y])
    return m_s


'''Espalhamento entre dois atributos'''


def scarter(v_x, v_y):
    if v_x.shape[0] != v_y.shape[0]:
        return None
    s = np.zeros(v_x.shape[0])

    mx = v_x.mean()
    my = v_y.mean()
    for el in range(v_x.shape[0]):
        s[el] = (v_x[el] - mx) * (v_y[el] - my)
    return sum(s)


def mean(m):
    if len(m.shape) == 1:
        return np.mean(m)
    return np.array([[np.mean(v) for v in m]]).T


def npc(ts, atr_tr, cls_tr):
    cls = []
    for el in range(len(cls_tr)):
        if el == 0:
            cls.append(cls_tr.T[0][el])
        else:
            if sum([cls_tr[el] == cls[i] for i in range(len(cls))]) == 0:
                cls.append(cls_tr.T[0][el])

    cls_p = np.array([cls, np.zeros(len(cls))])

    cols = []
    for i in range(atr_tr.shape[1] + 1):
        if i == atr_tr.shape[1]:
            cols.append('clas')
        else:
            cols.append(str(i))

    df = pd.DataFrame(np.vstack((atr_tr.T, cls_tr.T)).T, columns=cols)

    for cl in range(cls_p.shape[1]):
        c_i = df[df.clas == cls_p[0][cl]].values.T[:df.shape[1] - 1]
        m_i = mean(c_i).T[0]
        cls_p[1][cl] = np.sqrt(sum((ts - m_i) ** 2))

    cl_tst = 0
    for i in range(cls_p.shape[1]):
        if cls_p[1][i] < cls_p[1][cl_tst]:
            cl_tst = i

    return cls_p[0][cl_tst]


def lda(ts, atr_tr, cls_tr):
    ts = np.array([ts]).T
    cls = []
    for el in range(len(cls_tr)):
        if el == 0:
            cls.append(cls_tr.T[0][el])
        else:
            if sum([cls_tr[el] == cls[i] for i in range(len(cls))]) == 0:
                cls.append(cls_tr.T[0][el])

    cls_p = np.array([cls, np.zeros(len(cls))])

    cols = []
    for i in range(atr_tr.shape[1] + 1):
        if i == atr_tr.shape[1]:
            cols.append('clas')
        else:
            cols.append(str(i))

    df = pd.DataFrame(np.vstack((atr_tr.T, cls_tr.T)).T, columns=cols)

    if atr_tr.shape[1] == 1:
        cov = np.array([[np.cov(df.values.T[:df.shape[1] - 1])]])
    else:
        cov = np.cov(df.values.T[:df.shape[1] - 1])

    for cl in range(cls_p.shape[1]):
        c_i = df[df.clas == cls_p[0][cl]].values.T[:df.shape[1] - 1]
        m_i = mean(c_i)
        cls_p[1][cl] = ((ts - m_i).T @ np.linalg.inv(cov) @ (ts - m_i))[0][0]

    cl_tst = 0
    for i in range(cls_p.shape[1]):
        if cls_p[1][i] < cls_p[1][cl_tst]:
            cl_tst = i

    return cls_p[0][cl_tst]


def cda(atr, cls, nw):

    mf = mean(atr.T)

    classes = []

    for cl in range(cls.shape[0]):
        if cl == 0:
            classes.append(cls[cl][0])
        elif sum([cls[cl] == classes[i] for i in range(len(classes))]) == 0:
            classes.append(cls[cl][0])
    classes = np.array(classes)
    classes.sort()

    df = pd.DataFrame(np.vstack((atr.T, cls.T)).T, columns=i_atr)
    s_i = []
    m_i = []
    sb_i = []
    for cl in classes:
        c_i = df[df.clas == cl].values.T[:df.shape[1] - 1]
        s_i.append(mat_scarter(c_i))
        m_i.append(mean(c_i))
        sb_i.append((mean(c_i) - mf) @ (mean(c_i) - mf).T)
    sw = sum(s_i)
    sb = sum(sb_i)

    a_val, a_vet = eig(inv(sw) @ sb)

    return a_vet[:nw]


if __name__ == '__main__':

    data_f = get_data("data_dermato_03.txt")

    i_atr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
             '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', 'clas']
    '''    
    i_atr = ['1', '2', '3', '7', '9', '10', '14', '15', '16', '19', '20', '21', '22', '23', '24', '26', '28', '30',
             '31', '34', 'clas']
    i_atr = ['20', '22', 'clas']

    i_atr = ['2', '4', '5', '11', '13', '14', '15', '17', '18', '32', 'clas']
    i_atr = ['1', '2', '3', '7', '9', '10', '14', '15', '19', '20', '21', '22', '23', '24', '26', '28', '30', '31',
             'clas']

    i_atr = ['7', '15', '30', '31', 'clas']'''

    data = []

    for c in i_atr:
        if c == 'clas':
            data.append(data_f.T[data_f.shape[1] - 1])
        else:
            data.append(data_f.T[int(c) - 1])
    data = np.array(data).T

    data = f.z_score(data)
    #   data = f.normalize_min_max(data)
    atributos = data.T[:data.shape[1] - 1].T

    classes = data.T[data.shape[1] - 1:].T

    atributos, classes = shuffle(atributos, classes)

    tt = [70, 30]

    atributos_treino = atributos[:int(atributos.shape[0] * tt[0] / 100)]
    classes_treino = classes[:int(atributos.shape[0] * tt[0] / 100)]

    atributos_teste = atributos[int(atributos.shape[0] * tt[0] / 100):]
    classes_teste = classes[int(atributos.shape[0] * tt[0] / 100):]

    n_w = 5

    w = cda(atributos_treino, classes_treino, n_w)

    atributos_treino_p = (w @ atributos_treino.T).T

    atributos_teste_p = (w @ atributos_teste.T).T

    result = [npc(atributos_teste_p[t], atributos_treino_p, classes_treino) for t in range(atributos_teste_p.shape[0])]
    '''
    Resultado está variando muito, o acerto varia muito, é provável que a escolha de um conjunto de atributos melhor
    otimize o resultado
    '''
    print(f'{n_w} Projeções - Acerto: {((sum(classes_teste.T[0] == result) / classes_teste.shape[0]) * 100): .2f}%')
