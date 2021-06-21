import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random

with open('t10k-labels-idx1-ubyte','rb') as lbpath:
    actual_labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                            offset=8)
    
with open('t10k-images-idx3-ubyte','rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(actual_labels), 784)

pcs = PCA(n_components=185).fit(images)
t_images = pcs.transform(images)

kmeans = KMeans(n_clusters=10, random_state=0,init='k-means++').fit(t_images)
labels = kmeans.labels_
unique_labels = [0,1,2,3,4,5,6,7,8,9]

for i in range(10):
    cluster = np.asarray(actual_labels[np.where(labels==i)])
    percentage = [len(np.where(cluster==c)[0])/cluster.shape[0]*100 for c in unique_labels]
    plt.bar(unique_labels,percentage)
    plt.title(f'Cluster {i}')
    plt.xlabel('Class')
    plt.ylabel('percentage')
    plt.xticks
    plt.xticks(unique_labels)
    plt.show()