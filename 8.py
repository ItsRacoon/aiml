import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("D:\\lab\\aiml\\p8.csv")

X = dataset.iloc[:, :-1]
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}  
y = [label[c] for c in dataset.iloc[:, -1]]

plt.figure(figsize=(14,7))
colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y])

model = KMeans(n_clusters=3, random_state=3425).fit(X)
plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_])

gmm = GaussianMixture(n_components=3, random_state=3425).fit(X)
y_cluster_gmm = gmm.predict(X)
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm])

# Map the cluster labels to the actual class labels
def map_cluster_to_labels(cluster_labels, true_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, cluster_labels)
    return np.argmax(cm, axis=1)[cluster_labels]

kmeans_labels = map_cluster_to_labels(model.labels_, y)
gmm_labels = map_cluster_to_labels(y_cluster_gmm, y)

print('The accuracy score of K-Mean: ', metrics.accuracy_score(y, kmeans_labels))
print('The Confusion matrix of K-Mean:\n', confusion_matrix(y, kmeans_labels))

print('The accuracy score of EM: ', metrics.accuracy_score(y, gmm_labels))
print('The Confusion matrix of EM:\n', confusion_matrix(y, gmm_labels))

plt.show()
