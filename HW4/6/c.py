import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import silhouette_visualizer



data = pd.read_csv('ALS_train.csv').drop(['ID','SubjectID'],axis=1)
cols = ['ALSFRS_Total_range','trunk_range','hands_range','ALSFRS_Total_min','leg_range','mouth_range','trunk_min','mouth_min','respiratory_range','hands_min']
X = data[cols]
k=3
kmeans = KMeans(n_clusters=k, random_state=0,init='random')
silhouette_visualizer(kmeans, X, colors='yellowbrick')
print(pd.DataFrame(kmeans.cluster_centers_, columns=cols))

plt.bar(range(k),[np.count_nonzero(kmeans.labels_==i) for i in range(k)])
plt.xticks(range(k))
plt.xlabel('Cluster')
plt.show()
