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

pcs = PCA(n_components=185).fit(images)
t_images = pcs.transform(images)

colors = ['b','g','r','c','m','y','k','tab:gray','lime','orange']
unique_labels = [0,1,2,3,4,5,6,7,8,9]

kmeans = KMeans(n_clusters=10, random_state=0,init='k-means++').fit(t_images)
labels = kmeans.labels_
fig, ax = plt.subplots()
for i in range(len(unique_labels)):
    idxs = np.where(labels == unique_labels[i])
    ax.scatter(t_images[idxs,0], t_images[idxs,1], c=colors[i], label=unique_labels[i], s=10)
plt.title(f'k={10}')
ax.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(unique_labels)):
    idxs = np.where(labels == unique_labels[i])
    ax.scatter(t_images[idxs,0], t_images[idxs,1],t_images[idxs,2], c=colors[i],label=unique_labels[i],s=10)
plt.title(f'k={10}')
ax.legend()
plt.show()