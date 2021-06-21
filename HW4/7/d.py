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
centers4 = np.vstack((np.mean(t_images[np.in1d(labels,[1,3,5,7])],axis=0),
                    np.mean(t_images[np.in1d(labels,[2,4])],axis=0),
                    np.mean(t_images[np.in1d(labels,[6,8,10])],axis=0),
                    np.mean(t_images[np.in1d(labels,[9])],axis=0)))

centers7 = np.vstack((np.mean(t_images[np.in1d(labels,[1,3,5])],axis=0),
                    np.mean(t_images[np.in1d(labels,[2])],axis=0),
                    np.mean(t_images[np.in1d(labels,[4])],axis=0),
                    np.mean(t_images[np.in1d(labels,[6])],axis=0),
                    np.mean(t_images[np.in1d(labels,[7])],axis=0),
                    np.mean(t_images[np.in1d(labels,[9])],axis=0),
                    np.mean(t_images[np.in1d(labels,[8,10])],axis=0)))

centers10=np.vstack(np.mean(t_images[np.where(labels==i)],axis=0) for i in range(10))

for i,k in enumerate(k_arr):
    centers = [centers4,centers7,centers10]
    unique_labels = [0,1,2,3,4,5,6,7,8,9]
    unique_labels = unique_labels[:k]
    print(len(centers))
    kmeans = KMeans(n_clusters=k, random_state=0,init=centers[i]).fit(t_images)
    labels = kmeans.labels_
    fig, ax = plt.subplots()
    centers = kmeans.cluster_centers_ 
    for i in range(len(unique_labels)):
        idxs = np.where(labels == unique_labels[i])
        ax.scatter(t_images[idxs,0], t_images[idxs,1], c=colors[i], label=unique_labels[i], s=15)
        ax.scatter(centers[i,0], centers[i,1], c='gold', s=200, marker='*')
    plt.title(f'k={k}')
    ax.legend()
    plt.show()
