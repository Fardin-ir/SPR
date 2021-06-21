import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv('ALS_train.csv').drop(['ID','SubjectID'],axis=1)

X = data[['ALSFRS_Total_range','trunk_range','hands_range','ALSFRS_Total_min','leg_range','mouth_range','trunk_min','mouth_min','respiratory_range','hands_min']]

sse = []
for k in range(1,21):
    kmeans = KMeans(n_clusters=k, random_state=0,init='random').fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1,21),sse)
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(range(1,21))
plt.show()