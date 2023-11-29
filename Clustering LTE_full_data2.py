from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

data = pd.read_csv("Datasets//LTE_full_data.csv", index_col=None)

ranges = [(0, 299), (12000, 12299), (88400, 88699), (125000, 125299), (162891, 163079), (169970, 170080)]

selected_rows = pd.DataFrame()
for start, end in ranges:
    selected_rows = pd.concat([selected_rows, data.loc[start:end]])

selected_rows = selected_rows.replace('-', 0)

selected_rows['RSRP'] = selected_rows['RSRP'].astype(float)
selected_rows['RSRQ'] = selected_rows['RSRQ'].astype(float)

# Reset the index to start from 0 to 1500 directly (will need it later)
selected_rows = selected_rows.reset_index(drop=True)

# Extraire les colonnes spécifiées
X = selected_rows[['RSRP', 'RSRQ']]

# Normalisation 
scaler = StandardScaler()
X[['RSRP', 'RSRQ']] = scaler.fit_transform(X[['RSRP', 'RSRQ']])

# Création de l'objet KMeans avec un nombre de clusters variable
k_means = KMeans(n_clusters=5, random_state=0)

k_means.fit(X)

predicted_clusters = k_means.predict(X)

# Ajout des prédictions au DataFrame
X['Predicted_Cluster'] = predicted_clusters

# Extraction des données de chaque cluster
cluster_data = {}  

for cluster in range(5):  
    cluster_data[cluster] = X[X['Predicted_Cluster'] == cluster]

plt.figure(1)

colors = ['black', 'blue', 'green', 'yellow', 'purple']

for cluster in range(5):
    plt.scatter(cluster_data[cluster]['RSRQ'], cluster_data[cluster]['RSRP'], c=colors[cluster], label=f'Cluster {cluster}')

plt.xlabel('RSRQ')
plt.ylabel('RSRP')
plt.legend()
plt.title('Visualisation des clusters')
plt.show()

# Calcul des centroïdes
centroid_clusters = k_means.cluster_centers_


# Visualisation des centroïdes
plt.figure(2)

for cluster in range(5):
    plt.scatter(centroid_clusters[cluster, 0], centroid_clusters[cluster, 1], c=colors[cluster], label=f'Centroïde {cluster}')

plt.xlabel('RSRQ')
plt.ylabel('RSRP')
plt.legend()
plt.title('Visualisation des centroïdes')
plt.show()

    
'''--------------------variation du nbr de clusters de 2 a 4----------------------------------------------------------'''
for n in range(2, 5):
    kmeans = KMeans(n_clusters=n, n_init=5)
    kmeans.fit(X)

    predicted_clusters = kmeans.predict(X)

    # Add predictions to the DataFrame 'X'
    X[f'Predicted_Cluster_{n}'] = predicted_clusters
    
    # Visualize the clusters
    plt.figure()
    for cluster in range(n):
        cluster_mask = X[f'Predicted_Cluster_{n}'] == cluster
        plt.scatter(X[cluster_mask]['RSRQ'],X[cluster_mask]['RSRP'], )
    plt.ylabel('RSRP')
    plt.xlabel('RSRQ')
    plt.title(f'{n} clusters')

plt.show()

    
'''-----------------------------------inertie/silhouette---------------------------------------'''

inertia_values = []
silhouette_scores = []

for num_clusters in range(2, 16):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    kmeans.fit(X[['RSRP', 'RSRQ']])

    inertia = kmeans.inertia_
    inertia_values.append(inertia)

    silhouette_avg = silhouette_score(X[['RSRP', 'RSRQ']], kmeans.labels_, metric='euclidean')
    silhouette_scores.append(silhouette_avg)

plt.figure(3)
plt.plot(range(2, 16), inertia_values, marker='x')
plt.xlabel('clusters')
plt.ylabel('Inertie')

plt.plot(range(2, 16), silhouette_scores, marker='x')
plt.xlabel('Nclusters')
plt.ylabel('Score Silhouette')

plt.show()
