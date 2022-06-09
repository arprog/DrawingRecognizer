import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.backend import clear_session
from quickdraw import QuickDrawDataGroup
from sklearn.model_selection import KFold
from keras.metrics import Recall, Precision
from tensorflow_addons.metrics import F1Score
from keras.optimizers import Adam, Adagrad, Adamax, SGD, Nadam
from skmultilearn.model_selection import iterative_train_test_split
from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D, BatchNormalization)

tipos_desenhos = ['airplane', 'banana', 'bee', 'coffee cup', 'crab', 'guitar', 'hamburger', 'rabbit', 'truck', 'umbrella']

quantidade_tipos_desenhos = len(tipos_desenhos)
quantidade_desenhos = 500
largura_desenho = 130
altura_desenho = 130
batch = 50

def criar_modelo_cnn():
    model = Sequential()
    model.add(BatchNormalization(input_shape =(largura_desenho,altura_desenho,1)))
    model.add(Conv2D(16, (3,3), activation = 'relu'))
    model.add(Conv2D(16, (3,3), activation = 'relu'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation = 'softmax'))

    model.compile(optimizer = 'sgd',
                  loss = 'mean_squared_error',
                  metrics = ['accuracy', Recall(), Precision(), F1Score(num_classes=10)])

    return model

desenhos = list(map(lambda x: QuickDrawDataGroup(x, max_drawings=quantidade_desenhos), tipos_desenhos))

desenhos_cinzas = []
tipos_desenhos_cinzas = []

for i in range(quantidade_tipos_desenhos):
    for j in range(quantidade_desenhos):
        
        desenhos_do_tipo_analisado = desenhos[i]
        desenho = desenhos_do_tipo_analisado.get_drawing(index=j).image
        desenho_convertido_para_cinza = desenho.convert('L')
        
        desenho_convertido_para_cinza = np.array(desenho_convertido_para_cinza)
        desenho_convertido_para_cinza = cv.resize(desenho_convertido_para_cinza, (largura_desenho,altura_desenho))

        tipo_desenho_cinza = np.zeros(quantidade_tipos_desenhos)
        tipo_desenho_cinza[i] = 1

        desenhos_cinzas.append(desenho_convertido_para_cinza)
        tipos_desenhos_cinzas.append(tipo_desenho_cinza)

desenhos_cinzas = np.array(desenhos_cinzas).astype(np.float32)
tipos_desenhos_cinzas = np.array(tipos_desenhos_cinzas).astype(np.float32)

x_train, y_train, x_test, y_test = iterative_train_test_split(desenhos_cinzas, tipos_desenhos_cinzas, test_size=0.3)

acuracias = []
precisoes = []
recalls = []
total_tempos_treino = []
total_tempos_classificacao = []
perdas = []
lista_f1 = []

for i in range(30):
    
    model = criar_modelo_cnn()

    folds = KFold(n_splits=10, shuffle=False)
    
    acuracias_fold = []
    precisoes_fold = []
    recalls_fold = []
    tempos_treino_fold = []
    perdas_fold = []
    lista_f1_fold = []

    print("*------Execução: ", (i + 1), "------*")

    for train_index, _ in folds.split(x_train):

        horario_inicio = time()

        model.fit(x_train[train_index], y_train[train_index], epochs=10, batch_size=batch, validation_data=[x_test, y_test], shuffle=False)

        horario_termino = time()

        media_valor_acuracia = np.mean(model.history.history['val_accuracy'])
        media_valor_perda = np.mean(model.history.history['val_loss'])
        media_valor_recall = np.mean(model.history.history['val_recall'])
        media_valor_precisao = np.mean(model.history.history['val_precision'])
        media_valor_f1 = np.mean(model.history.history['val_f1_score'])

        acuracias_fold.append(media_valor_acuracia)
        perdas_fold.append(media_valor_perda)
        precisoes_fold.append(media_valor_precisao)
        recalls_fold.append(media_valor_recall)
        tempos_treino_fold.append(horario_termino - horario_inicio)
        lista_f1_fold.append(media_valor_f1)

    acuracias.append(np.mean(acuracias_fold))
    perdas.append(np.mean(perdas_fold))
    precisoes.append(np.mean(precisoes_fold))
    recalls.append(np.mean(recalls_fold))
    total_tempos_treino.append(np.mean(tempos_treino_fold))
    lista_f1.append(np.mean(lista_f1_fold))

    tempos_classificacao = []
    
    for feature in x_test:

        horario_inicio = time()

        model.predict(feature.reshape(1, feature.shape[0], feature.shape[1], 1))

        horario_termino = time()

        tempos_classificacao.append((horario_termino - horario_inicio) * 1000)

    total_tempos_classificacao.append(np.mean(tempos_classificacao))

    clear_session()

dataframe = pd.DataFrame(data={"Accuracy": acuracias,
                            "Precision": precisoes,
                            "Recall": recalls,
                            "Fit Time": total_tempos_treino,
                            "Time Classification": total_tempos_classificacao,
                            "F1score": lista_f1,
                            "Loss": perdas})

dataframe.to_excel("rnn.xlsx", index=False)

model.save('cnn.h5')