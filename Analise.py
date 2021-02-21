"""coding: utf-8"""
import funcoes as f
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = f.get_data("data_dermato_03.txt")

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
         "melanin_incontinence": data.T[12],
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
         "clas": data.T[34]
         }
    )
    '''

    atributos = data.T[:34].T
    classes = data.T[34:].T

    atributos, classes = shuffle(atributos, classes, random_state=0)

    vet_media_atr = f.mean(atributos)
    vet_variancia_atr = f.variance(atributos)
    mat_covariancia = f.mat_covarience(atributos)
    mat_correlacao = f.mat_corelation(data)

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
