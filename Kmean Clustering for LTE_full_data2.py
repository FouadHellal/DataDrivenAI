# Import necessary libraries
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Read data from CSV file
data = pd.read_csv("Datasets/LTE_full_data.csv", index_col=None)

# Define ranges for specific rows in the dataset
ranges = [(0, 299), (12000, 12299), (88400, 88699), (125000, 125299), (162891, 163079), (169970, 170080)]

# Select specific rows based on the defined ranges
selected_rows = pd.DataFrame()
for start, end in ranges:
    selected_rows = pd.concat([selected_rows, data.loc[start:end]])

# Replace '-' with 0 in selected rows and convert columns to float
selected_rows = selected_rows.replace('-', 0)
selected_rows['RSRP'] = selected_rows['RSRP'].astype(float)
selected_rows['RSRQ'] = selected_rows['RSRQ'].astype(float)

# Reset the index to start from 0 to 1500 directly (will need it later)
selected_rows = selected_rows.reset_index(drop=True)

# Extract specified columns for clustering
X = selected_rows[['RSRP', 'RSRQ']]

# Standardize the data
scaler = StandardScaler()
X[['RSRP', 'RSRQ']] = scaler.fit_transform(X[['RSRP', 'RSRQ']])

# Create a KMeans object with a variable number of clusters
k_means = KMeans(n_clusters=5, random_state=0)
k_means.fit(X)
predicted_clusters = k_means.predict(X)

# Add predictions to the DataFrame
X['Predicted_Cluster'] = predicted_clusters

# Visualize the clusters
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
plt.title('Visualization of Clusters')
plt.show()

# Calculate cluster centroids
centroid_clusters = k_means.cluster_centers_

# Visualize cluster centroids
plt.figure(2)
for cluster in range(5):
    plt.scatter(centroid_clusters[cluster, 0], centroid_clusters[cluster, 1], c=colors[cluster], label=f'Centroid {cluster}')

plt.xlabel('RSRQ')
plt.ylabel('RSRP')
plt.legend()
plt.title('Visualization of Centroids')
plt.show()

# Variation of the number of clusters from 2 to 4
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
        plt.scatter(X[cluster_mask]['RSRQ'],X[cluster_mask]['RSRP'])
    plt.ylabel('RSRP')
    plt.xlabel('RSRQ')
    plt.title(f'{n} clusters')

plt.show()

# Inertia/Silhouette analysis
inertia_values = []
silhouette_scores = []

for num_clusters in range(2, 16):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    kmeans.fit(X[['RSRP', 'RSRQ']])

    inertia = kmeans.inertia_
    inertia_values.append(inertia)

    silhouette_avg = silhouette_score(X[['RSRP', 'RSRQ']], kmeans.labels_, metric='euclidean')
    silhouette_scores.append(silhouette_avg)

# Plotting Inertia and Silhouette scores
plt.figure(3)
plt.plot(range(2, 16), inertia_values, marker='x')
plt.xlabel('Clusters')
plt.ylabel('Inertia')

plt.plot(range(2, 16), silhouette_scores, marker='x')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.show()
