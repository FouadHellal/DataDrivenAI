import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
OMP_NUM_THREADS=3

'''-------------------------------------------------------------------------------------------'''
file= "Datasets/DataTelecom.csv"
data = pd.read_csv(file)


x = data[['Frequency', 'SNR', 'Amplitude']]
y =data['Class']

# nombre de clusters = 5
kmeans = KMeans(n_clusters=5,n_init=10)

# Apprentissage ta3 data X
kmeans.fit(x)
labels=kmeans.labels_
# Prédiction des clusters pour Xi
predicted_clusters = kmeans.predict(x)

# Ajout des prédictions li jawna au DataFrame
data['Predicted_Cluster'] = predicted_clusters

#REGARDE LE DATA FRAME POUR COMPARER !
# Calcul du pourcentage d'exactitude
Exact = adjusted_rand_score(y, predicted_clusters)
print("Pourcentage d'exactitude",Exact*100,"%")

kmeans.inertia_ #la somme entre les pts d'un cluster et son centroid
kmeans.score(x)

"""3.2)"""
# Extraction des données des clusters

cluster_data = {} #Un dictionnaire pour stocker les data frames ta3 chaque cluster


for cluster in range(5):  
    cluster_data[cluster] = data[data['Predicted_Cluster'] == cluster]


for cluster in range(5):
    plt.scatter(cluster_data[cluster]['Frequency'], cluster_data[cluster]['SNR'])
plt.figure(1)
plt.xlabel('Frequency')
plt.ylabel('SNR')
plt.show()

'''--------------------Cooking--------------------------------------------------------------'''

cluster_data = {}  # Un dictionnaire pour stocker les data frames de chaque cluster

for cluster in range(5):
    cluster_data[cluster] = data[data['Predicted_Cluster'] == cluster]
    
    plt.figure()
    
    plt.scatter(cluster_data[cluster]['Frequency'], cluster_data[cluster]['SNR'])
    
    #pour maintenir l'echelle kamla, on prends les max et les min ta3 freq et snr bah ybano bien clusters
    plt.xlim(data['Frequency'].min(), data['Frequency'].max())
    plt.ylim(data['SNR'].min(), data['SNR'].max())
    plt.xlabel('Frequency')
    plt.ylabel('SNR')
    plt.title(f'Cluster {cluster + 1}')
    plt.show()

'''-----------------------------------------------------------------------------------------'''

# Calcul des centroïdes

centroidClusters = kmeans.cluster_centers_

# Visualisation des centroïdes en utilisant scatter
plt.figure(2)

for cluster in range(5):
    plt.scatter(centroidClusters[cluster, 0], centroidClusters[cluster, 1])

plt.xlabel('Frequency')
plt.ylabel('SNR')
plt.title('The damn centroids')
plt.show()

"""-----------------------------4)Clustering avec Normalisation-------------------------------------------"""


file = "C:/Users/helfo/Downloads/DataTelecom.csv"
data = pd.read_csv(file)

x = data[['Frequency', 'SNR', 'Amplitude']]
y = data['Class']

# Normalisation Min-Max
scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x)


for n in range(2,5):

    kmeans = KMeans(n_clusters=n, n_init=5)
    kmeans.fit(x_normalized)

    predicted_clusters = kmeans.predict(x_normalized)

    Exact = accuracy_score(y, predicted_clusters)
    print(f"clusters : {n}")
    print("Pourcentage d'exactitude",Exact*100,"%")


    # Ajout des prédictions au DataFrame
    data[f'Predicted_Cluster_{n}'] = predicted_clusters
    
    # Visualisation des clusters
    plt.figure()
    for cluster in range(n):
        cluster_mask = data[f'Predicted_Cluster_{n}'] == cluster
        plt.scatter(x_normalized[cluster_mask, 0], x_normalized[cluster_mask, 1])
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Normalized SNR')
    plt.title(f'{n} clusters')

plt.show()


"""-----------------------------5)la méthode Elbow / Silhouette-------------------------------------------"""

inertia = []
silhouette_scores = []


for num_clusters in range(2, 16):
    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(x_normalized)
    
    #inertie
    inertie = k_means.inertia_
    
    #silhouette
    #silhouette = sum(np.min(cdist(x_normalized, k_means.cluster_centers_,'euclidean'), axis=1)) / Xi_normalized.shape[0] WTF
    silhouette = silhouette_score(x_normalized, k_means.labels_)
    
    inertia.append(inertie)
    silhouette_scores.append(silhouette)

plt.figure(3)
plt.plot(range(2, 16), inertia, marker='o')
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.show()

plt.figure(4)
plt.plot(range(2, 16), silhouette_scores, marker='o')
plt.xlabel("Nombre de clusters")
plt.ylabel("Score Silhouette")
plt.show()
