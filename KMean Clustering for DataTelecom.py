# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Load the dataset
file = "Datasets/DataTelecom.csv"
data = pd.read_csv(file)

# Extract features and target variable
x = data[['Frequency', 'SNR', 'Amplitude']]
y = data['Class']

# Number of clusters = 5
kmeans = KMeans(n_clusters=5, n_init=10)

# Train the model on features X
kmeans.fit(x)
labels = kmeans.labels_
# Predict clusters for X
predicted_clusters = kmeans.predict(x)

# Add predictions to the DataFrame
data['Predicted_Cluster'] = predicted_clusters

# Calculate accuracy percentage
exact = adjusted_rand_score(y, predicted_clusters)
print("Accuracy Percentage:", exact * 100, "%")

# Get inertia and score
kmeans_inertia = kmeans.inertia_
kmeans_score = kmeans.score(x)

# Extract data from clusters
cluster_data = {}
for cluster in range(5):
    cluster_data[cluster] = data[data['Predicted_Cluster'] == cluster]

# Visualize clusters in 2D
for cluster in range(5):
    plt.scatter(cluster_data[cluster]['Frequency'], cluster_data[cluster]['SNR'])
plt.figure(1)
plt.xlabel('Frequency')
plt.ylabel('SNR')
plt.show()

'''--------------------Cooking--------------------------------------------------------------'''

cluster_data = {}

# Visualize clusters with normalized scales
for cluster in range(5):
    cluster_data[cluster] = data[data['Predicted_Cluster'] == cluster]

    plt.figure()

    plt.scatter(cluster_data[cluster]['Frequency'], cluster_data[cluster]['SNR'])

    plt.xlim(data['Frequency'].min(), data['Frequency'].max())
    plt.ylim(data['SNR'].min(), data['SNR'].max())
    plt.xlabel('Frequency')
    plt.ylabel('SNR')
    plt.title(f'Cluster {cluster + 1}')
    plt.show()

'''-----------------------------------------------------------------------------------------'''

# Calculate centroids
centroidClusters = kmeans.cluster_centers_

# Visualize centroids using scatter plot
plt.figure(2)
for cluster in range(5):
    plt.scatter(centroidClusters[cluster, 0], centroidClusters[cluster, 1])

plt.xlabel('Frequency')
plt.ylabel('SNR')
plt.title('Centroids')
plt.show()

"""-----------------------------Clustering with Normalization-------------------------------------------"""

x = data[['Frequency', 'SNR', 'Amplitude']]
y = data['Class']

# Min-Max normalization
scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x)

for n in range(2, 5):
    kmeans = KMeans(n_clusters=n, n_init=5)
    kmeans.fit(x_normalized)

    predicted_clusters = kmeans.predict(x_normalized)

    # Calculate accuracy percentage
    exact = accuracy_score(y, predicted_clusters)
    print(f"Clusters: {n}")
    print("Accuracy Percentage:", exact * 100, "%")

    # Add predictions to the DataFrame
    data[f'Predicted_Cluster_{n}'] = predicted_clusters

    # Visualize clusters
    plt.figure()
    for cluster in range(n):
        cluster_mask = data[f'Predicted_Cluster_{n}'] == cluster
        plt.scatter(x_normalized[cluster_mask, 0], x_normalized[cluster_mask, 1])
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Normalized SNR')
    plt.title(f'{n} clusters')

plt.show()

"""-----------------------------With Elbow Method / Silhouette-------------------------------------------"""

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
plt.figure(3)
plt.plot(range(2, 16), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Plotting Silhouette Scores
plt.figure(4)
plt.plot(range(2, 16), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method")
plt.show()
