# Dermatologia

####funções.py
Contém algumas funções necessárias
- Covariância (Matriz de covariância)
- Correlação (Matriz de correlação)


####Analise.py
faz a análize prévia da base de dados:
- Vetor médio (média de cada atributo).
- Vetor de variâncias (variância de cada atributo).
- Matriz de covariância
- Matriz de correlação

## Matriz de Correlação
A matriz de correlação indica o quão correlacionados os atributos estão
Os atributos 6, 8, 12, 25, 27, 29 e 33 estão correlacionados entre si.
Os atributos 7, 30 e 31 também estão correlacionados entre si.

## Classificadores Baysianos
### Naive Bayes
Ultilizando as probabilidades de um vetor de atribututos pertencer a cada classe no conjunto
de treino, obteve-se u erro médio de 0.55% ou seja 99.45% de acerto.

### LDA
Esse classificador obteve uma taxa média de erro de 1.92%, ou seja 98.8% de acerto.
### QDA
Obeteve uma taxa média de erro variável, entretanto, nas rodadas do k-fold, apresentam casos de taxas de acerto por 
volta de 90%.

### CDA
Resultado está variando muito, o acerto varia muito, é provável que a escolha de um conjunto de atributos melhor 
otimize o resultado

### OBSERVAÇÔES
####O Naive Bayes obteve uma maior taxa de acerto.