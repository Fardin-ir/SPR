import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random

with open('t10k-labels-idx1-ubyte','rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                            offset=8)
    
with open('t10k-images-idx3-ubyte','rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(labels), 784)

pcs = PCA(n_components=185).fit(images)
t_images = pcs.transform(images)

kmeans = KMeans(n_clusters=10, random_state=0,init='k-means++').fit(t_images)
labels = kmeans.labels_


for c in range(10):
    fig, axs = plt.subplots(2, 5)
    fig.tight_layout()
    cluster = images[np.where(labels==c)]
    for i in range(2):
        for j in range(5):
            x = random.randint(0,cluster.shape[0])
            axs[i,j].imshow(cluster[x,:].reshape(28,28),cmap='gray')
            axs[i,j].set_title(f'cluster {c}',size=10)
    plt.show()
