import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report, ConfusionMatrixDisplay
import tensorflow as tf # importer tensorflow
from tensorflow import keras # importer Keras
from keras.models import Sequential # Créer le modèle multicouche
from keras import optimizers # importer les optimiseurs : SGD, Adam, RMSprop
from keras.utils import to_categorical # transformer les labels y en vecteurs canoniques
from tensorflow.keras.layers import Conv1D, LeakyReLU, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D

LTE_data = pd.read_csv("C:/Users/helfo/Downloads/LTE_new_data.csv", index_col=None)
Base=LTE_data[['RSRP', 'RSRQ','RSSI', 'SNR','DL_bitrate', 'UL_bitrate', 'path' ]]
Base=Base.replace("-", 0)
Appr=Base.drop('path', axis=1)
classe2=Base['path'] #Extrait la colonne path (variable cible)
scaler2 = StandardScaler()
A=scaler2.fit_transform(Appr) #Standardisation 
le= LabelEncoder() #convertit les str en valeurs numériques
classe = le.fit_transform(classe2)

App, test, y_train, y_test = train_test_split(A, classe, test_size=0.2,
                                                            random_state=42)
L_App = to_categorical(y_train)
L_test = to_categorical(y_test)

'''-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._'''
#3 Simple CNN layers with Flatten then dense
def cnn(kpi, classes):
    model = Sequential()
    
    model.add(Conv1D(32, kernel_size=3, input_shape=(kpi, 1), padding='same'))
    #padding='same' ensures that the spatial dimensions of the input remain the same after the convolution operation, preventing information loss at the borders of the input
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D())
    
    model.add(Conv1D(64, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D())
    
    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Dense(60))    
    model.add(Dense(classes, activation='softmax'))
    
    return model

#3 Simple CNN layers then BatchNormalization with Flatten then dense
def cnn_batch_norm(kpi, classes):
    model = Sequential()
    
    model.add(Conv1D(32, kernel_size=3, input_shape=(kpi, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D())

    
    model.add(Conv1D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D())
    
    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Flatten())
    model.add(Dropout(rate=0.15))
  
    model.add(Dense(128))    
    model.add(Dense(60))    
    model.add(Dense(classes, activation='softmax'))
    
    return model

#3 Simple CNN layers then BatchNormalization with GAP instead of flatten-dense
def cnn_GAP_fct(kpi, classes, fct):
    model = Sequential()
    
    model.add(Conv1D(32, kernel_size=3, input_shape=(kpi, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D())

    model.add(Conv1D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D())

    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    
    # Global Average Pooling
    model.add(GlobalAveragePooling1D())  

    model.add(Dense(classes, activation=fct))
    
    return model

'''-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._'''
#TRY YOUR MODEL HERE :
def evaluate_model(kpi,classes,fct,batch,i):
    
    if i==1:
        model = cnn(kpi, classes)
    elif i==2:
        model = cnn_batch_norm(kpi, classes)
    elif i==3:
        model = cnn_GAP_fct(kpi, classes,fct)
    
    #architecture du modèle
    model.summary()
    
    # COMPILER
    # On spécifie comment le modèle doit apprendre
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
                  metrics=['accuracy'])
    
    history = model.fit(
        App,                     # Données d'Apprentissage
        L_App,                   # Étiquettes d'Apprentissage
        epochs=500,              # Nombre d'époques d'entraînement
        batch_size=batch,           # Taille du lot (nombre d'échantillons par mise à jour des poids)
        steps_per_epoch=len(App) / batch,  # Nombre d'étapes par époque (nombre total d'échantillons divisé par la taille du lot)
    )
    
    # Évaluation finale sur les données de test
    _, final_accuracy = model.evaluate(test, L_test)
    print("Accuracy des données de test:", final_accuracy*100,'%')
    
    pred=model.predict(test)
    y_pred = np.argmax(pred, axis=1)
    
    # de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()

ii=input('choisissez votre cnn :')
evaluate_model(kpi=6,classes=5, fct='softmax', batch=32,i=int(ii))
