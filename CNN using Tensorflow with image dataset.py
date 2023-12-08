'''-----------------------IMAGES POTATO ----------------------------------------------'''
import numpy as np
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
import tensorflow as tf # importer tensorflow
from tensorflow import keras # importer Keras
from keras.models import Sequential # Créer le modèle multicouche
from keras import optimizers # importer les optimiseurs : SGD, Adam, RMSprop
from keras.utils import to_categorical # transformer les labels y en vecteurs canoniques
from tensorflow.keras.layers import Conv2D,GlobalAveragePooling2D,AveragePooling2D,MaxPooling2D,Conv1D, LeakyReLU, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, BatchNormalization, GlobalAveragePooling1D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

Liste = [
    "Datasets/Dataset/Potato_Late_blight",
    "Datasets/Dataset/Potato_Early_blight",
    "Datasets/Dataset/Potato_healthy"
    ]

# Fonction pour générer les caractéristiques
def preTrait(folder, data):
    for pic in os.listdir(folder):
        img = plt.imread(os.path.join(folder, pic))
        ima=cv2.resize(img, (40,40), interpolation = cv2.INTER_AREA)
        data.append(ima)
    return data

# Création de la base de données
Base = []
for i in range(len(Liste)):
    preTrait(Liste[i], Base)  

Base = np.array(Base)

# Création des étiquettes
labels = []
for i in range(len(Liste)):
    classe_dir = Liste[i]
    image_files = [file for file in os.listdir(classe_dir) if '.' in file]
    class_index_labels = [i] * len(image_files)
    labels.extend(class_index_labels)

# Convertir la liste en un tableau NumPy
labels = np.array(labels)
#split
App, test, y_train, y_test = train_test_split(Base, labels,
                                              test_size=0.2,
                                              random_state=42)
#canonisation
L_App = to_categorical(y_train)
L_test = to_categorical(y_test)

'''-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._-._.-._'''

def modele2(input_shape, num_classes,n_filter):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(n_filter, n_filter), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, kernel_size=(n_filter, n_filter)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, kernel_size=(n_filter, n_filter)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Dense(60))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def modele2_GAP(input_shape, num_classes,n_filter):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(n_filter, n_filter), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, kernel_size=(n_filter, n_filter)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, kernel_size=(n_filter, n_filter)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(GlobalAveragePooling2D())  

    model.add(Dense(num_classes, activation='relu'))
    
    return model

input_shape = (40, 40, 3)  # Modifier la taille en fonction de vos images
num_classes = 3  # Nombre de classes dans votre ensemble de données
model = modele2_GAP(input_shape, num_classes,5)
#model = modele2_GAP(input_shape, num_classes,5)

#architecture du modèle
model.summary()

#COMPILER
# On spécifie comment le modèle doit apprendre
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
              metrics=['accuracy'])


history = model.fit(
    App,                     # Données d'Apprentissage
    L_App,                   # Étiquettes d'Apprentissage
    epochs=200,              # Nombre d'époques d'entraînement
    batch_size=16,           # Taille du lot (nombre d'échantillons par mise à jour des poids)
    steps_per_epoch=len(App) /16,  # Nombre d'étapes par époque (nombre total d'échantillons divisé par la taille du lot)
)
# Évaluation finale sur les données de test
_, final_accuracy = model.evaluate(test, L_test)
print("Accuracy des données de test:", final_accuracy*100 ,'%')

pred=model.predict(test)

# Convertir les prédictions en classes
y_pred = np.argmax(pred, axis=1)

# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
le = LabelEncoder()
le.fit(labels)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()

# Afficher le rapport de classification
report = classification_report(y_test, y_pred, target_names=None)
print(report)
