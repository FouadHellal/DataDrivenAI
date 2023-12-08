
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
#import os
#os.environ['LOKY_MAX_CPU_COUNT'] = '4'
#OMP_NUM_THREADS=3


data = pd.read_csv("Datasets/House_prediction.txt")
x = data

# nombre de clusters = 5
kmeans = KMeans(n_clusters=5,n_init=10)

# Apprentissage ta3 data X
kmeans.fit(x)
labels=kmeans.labels_
# Prédiction des clusters pour Xi
predicted_clusters = kmeans.predict(x)

# Ajout des prédictions li jawna au DataFrame
data['Predicted_Cluster'] = predicted_clusters
cluster_data = {} #Un dictionnaire pour stocker les data frames ta3 chaque cluster


'''---------------------------Normalisation Min-Max---------------------------------------'''

scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x)

for n in range(2,5):

    kmeans = KMeans(n_clusters=n, n_init=5)
    kmeans.fit(x_normalized)

    predicted_clusters = kmeans.predict(x_normalized)

    #accuracy = (predicted_clusters == y).mean()

    print(f"clusters : {n}")
    #print(f"Précision : {accuracy:.2f}\n")

    # Ajout des prédictions au DataFrame
    data[f'Predicted_Cluster_{n}'] = predicted_clusters
    
    # Visualisation des clusters
    plt.figure()
    
    for cluster in range(n):
        cluster_mask = data[f'Predicted_Cluster_{n}'] == cluster
        plt.scatter(x_normalized[cluster_mask, 0], x_normalized[cluster_mask, 1])
    plt.title(f'{n} clusters')

plt.show()


"""----------------------------- méthode Elbow / Silhouette-------------------------------------------"""

inertia = []
silhouette_scores = []


for num_clusters in range(2, 16):
    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(x_normalized)
    
    #inertie)
    inertie = k_means.inertia_
    #silhouette
    #silhouette = sum(np.min(cdist(x_normalized, k_means.cluster_centers_,'euclidean'), axis=1)) / Xi_normalized.shape[0] WTF
    silhouette = silhouette_score(x_normalized, k_means.labels_)
    
    inertia.append(inertie)
    silhouette_scores.append(silhouette)

plt.figure()
plt.plot(range(2, 16), inertia, marker='o')
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.show()

plt.figure()
plt.plot(range(2, 16), silhouette_scores, marker='o')
plt.xlabel("Nombre de clusters")
plt.ylabel("Score Silhouette")
plt.show()


    

