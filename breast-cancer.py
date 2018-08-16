import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split

previsoresTrain, previsoresTest, classTrain, classTest = train_test_split(previsores, classe, test_size=0.25)

'''print(len(previsoresTrain))
print(len(previsoresTest))
print(len(classTrain))
print(len(classTest))'''

import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
'''Quantidade de Units: numero de atributos + numeros de saida / 2.
Activation é a funçao de ativacao. relu é melhor pra DL! || Kernel_initializer é a inicializaçao dos pesos.
Input_Dim é a quantidade de elementos na camada de entrada! || Para mais detalhes: www.keras.io '''

classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform')) #Tira o INPUT DIM!

classificador.add(Dense(units = 1, activation = 'sigmoid')) #Essa é a camada de saída.

#Agora temos uma rede com 2 camadas ocultas.

otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)

classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
 #Caso resultados nao sejam bons, testar outros otimizadores!

classificador.fit(previsoresTrain, classTrain, batch_size = 10, epochs = 100)

pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(previsoresTest)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classTest, previsoes)
matriz = confusion_matrix(classTest, previsoes)

resultado = classificador.evaluate(previsoresTest, classTest)
