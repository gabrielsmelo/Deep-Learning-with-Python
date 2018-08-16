import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def trunc(num, digits):
   sp = str(num).split('.')
   return '.'.join([sp[0], sp[:digits]])

def criarRede():
    rna = Sequential()
    rna.add(Dense(units = 16, activation = 'relu',
     kernel_initializer = 'random_uniform', input_dim = 30))
    rna.add(Dropout(0.2))
    rna.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
    rna.add(Dropout(0.2))
    rna.add(Dense(units = 1, activation = 'sigmoid'))

    otm = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    rna.compile(optimizer = otm, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

    return rna #Rede Neural Artificial!

classificador = KerasClassifier(build_fn = criarRede,
                                 epochs = 100,
                                 batch_size = 10)

resultados = cross_val_score(estimator = classificador, X = previsores, y = classe,
                                cv = 10, scoring = 'accuracy' )

print('\n', resultados)

media = resultados.mean()
desvio = resultados.std()
print('\n A Média desta rede é: ', media)
media = media * 100
print('\nQue, diga-se de passagem, significa {}{} de acerto.\n'.format(round(media, 2), '%'))
print('O desvio padrao é {} \n'.format(desvio))
#O desvio mostra se tá acontecendo overfitting.