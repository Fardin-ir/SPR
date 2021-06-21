import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from yellowbrick.cluster import silhouette_visualizer,SilhouetteVisualizer

data = pd.read_csv('ALS_train.csv').drop(['ID','SubjectID'],axis=1)
X = data[['ALSFRS_Total_range','trunk_range','hands_range','ALSFRS_Total_min','leg_range','mouth_range','trunk_min','mouth_min','respiratory_range','hands_min']]


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


clustering1 = AgglomerativeClustering(n_clusters=3,linkage='single',compute_distances=True).fit(X)
clustering2 = AgglomerativeClustering(n_clusters=3,linkage='complete',compute_distances=True).fit(X)
clustering3 = AgglomerativeClustering(n_clusters=3,linkage='average',compute_distances=True).fit(X)

plot_dendrogram(clustering1, truncate_mode='level', p=3)
plt.show()
model = SilhouetteVisualizer(clustering1, colors='yellowbrick')
model.fit(X)
model.show()

plot_dendrogram(clustering2, truncate_mode='level', p=3)
plt.show()
model = SilhouetteVisualizer(clustering2, colors='yellowbrick')
model.fit(X)
model.show()

plot_dendrogram(clustering3, truncate_mode='level', p=3)
plt.show()
model = SilhouetteVisualizer(clustering3, colors='yellowbrick')
model.fit(X)
model.show()
