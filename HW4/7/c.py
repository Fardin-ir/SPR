import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

with open('t10k-labels-idx1-ubyte','rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                            offset=8)
    
with open('t10k-images-idx3-ubyte','rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(labels), 784)

colors = ['b','g','r','c','m','y','k','tab:gray','lime','orange']

pcs = PCA(n_components=2).fit(images)
eigenvalues = pcs.explained_variance_
t_images = pcs.transform(images)

k_arr = [4,7,10]
for k in k_arr:
    unique_labels = [0,1,2,3,4,5,6,7,8,9]
    unique_labels = unique_labels[:k]
    kmeans = KMeans(n_clusters=k, random_state=0,init='random').fit(t_images)
    labels = kmeans.labels_
    fig, ax = plt.subplots()
    centers = kmeans.cluster_centers_ 
    for i in range(len(unique_labels)):
        idxs = np.where(labels == unique_labels[i])
        ax.scatter(t_images[idxs,0], t_images[idxs,1], c=colors[i], label=unique_labels[i], s=10)
        ax.scatter(centers[i,0], centers[i,1], c='gold', s=200, marker='*')
    plt.title(f'k={k}')
    ax.legend()
    plt.show()
