from sklearn.cluster import KMeans 
import numpy as np 
import matplotlib.pyplot as plt 

# Data points
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8]) 
x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3]) 

# Initial scatter plot of the dataset
plt.figure(figsize=(8, 6))
plt.xlim([0, 10]) 
plt.ylim([0, 10]) 
plt.title('Dataset') 
plt.scatter(x1, x2, color='black', marker='o')  # Plot data points in black
plt.xlabel('x1')
plt.ylabel('x2')
plt.show() 

# Prepare data for clustering
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# KMeans algorithm
K = 3  # Number of clusters
kmeans_model = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)  # Fit the model

# Plot clustered data
plt.figure(figsize=(8, 6))
colors = ['b', 'g', 'r']  # Colors for different clusters
markers = ['o', 'v', 's']  # Markers for different clusters

# Plot each point with corresponding color and marker based on cluster label
for i, label in enumerate(kmeans_model.labels_): 
    plt.plot(x1[i], x2[i], color=colors[label], marker=markers[label], ls='None')

plt.xlim([0, 10]) 
plt.ylim([0, 10]) 
plt.title('KMeans Clustering')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
