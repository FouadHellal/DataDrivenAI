# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Read data from CSV file
data = pd.read_csv("Datasets/House_prediction.txt")
x = data

# Number of clusters = 5
kmeans = KMeans(n_clusters=5, n_init=10)

# Train the model on data X
kmeans.fit(x)
labels = kmeans.labels_
# Predict clusters for Xi
predicted_clusters = kmeans.predict(x)

# Add predictions to the DataFrame
data['Predicted_Cluster'] = predicted_clusters
cluster_data = {}  # A dictionary to store data frames for each cluster


'''---------------------------Min-Max Normalization---------------------------------------'''

scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x)

# Iterate over different cluster numbers (2 to 4)
for n in range(2, 5):
    kmeans = KMeans(n_clusters=n, n_init=5)
    kmeans.fit(x_normalized)

    predicted_clusters = kmeans.predict(x_normalized)

    # Add predictions to the DataFrame
    data[f'Predicted_Cluster_{n}'] = predicted_clusters

    # Visualize the clusters
    plt.figure()
    for cluster in range(n):
        cluster_mask = data[f'Predicted_Cluster_{n}'] == cluster
        plt.scatter(x_normalized[cluster_mask, 0], x_normalized[cluster_mask, 1])
    plt.title(f'{n} clusters')

plt.show()

"""----------------------------- Elbow Method / Silhouette Score-------------------------------------------"""

inertia = []
silhouette_scores = []

# Iterate over different cluster numbers (2 to 15)
for num_clusters in range(2, 16):
    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(x_normalized)

    # Inertia
    inertie = k_means.inertia_

    # Silhouette Score
    silhouette = silhouette_score(x_normalized, k_means.labels_)

    inertia.append(inertie)
    silhouette_scores.append(silhouette)

# Plotting the Elbow Method
plt.figure()
plt.plot(range(2, 16), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Plotting Silhouette Scores
plt.figure()
plt.plot(range(2, 16), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method")
plt.show()
