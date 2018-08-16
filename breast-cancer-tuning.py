#Esse programa roda por umas 8h. Rode no LAVID!


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    rna = Sequential()
    rna.add(Dense(units = neurons, activation = activation,
     kernel_initializer = kernel_initializer, input_dim = 30))
    rna.add(Dropout(0.2))
    rna.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    rna.add(Dropout(0.2))
    rna.add(Dense(units = 1, activation = 'sigmoid')) #Nao faz sentido NÃO USAR a sigmoid.

    # otm = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    rna.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy']) #Estamos trabalhando com class binaria
    #Então pode ser essa metrica mesmo

    return rna #Rede Neural Artificial!

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 30],
            'epochs': [50, 100],
            'optimizer': ['adam', 'sgd'], #adam tras resultados melhores
            'loss': ['binary_crossentropy', 'hinge'], #binary tras resultados melhores
            'kernel_initializer': ['random_uniform', 'normal'],
            'activation': ['relu', 'tanh'],
            'neurons': [16, 8]}

gridSearch = GridSearchCV(estimator = classificador, param_grid = parametros,
                scoring = 'accuracy', cv = 5)

gridSearch = gridSearch.fit(previsores, classe)
bestParams = gridSearch.best_params_
bestPredict = gridSearch.best_score_

print('\n',bestParams, '\n')
print('\n', bestPredict, '\n')

melhores = open('best.txt', 'w')
melhores.write(bestParams)
melhores.write('\n')
melhores.write(bestPredict)
melhores.close()
