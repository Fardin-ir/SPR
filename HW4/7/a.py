import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

with open('t10k-labels-idx1-ubyte','rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                            offset=8)
    
with open('t10k-images-idx3-ubyte','rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(labels), 784)

colors = ['b','g','r','c','m','y','k','tab:gray','lime','orange']

pcs = PCA(n_components=20).fit(images)
eigenvalues = pcs.explained_variance_
t_images = pcs.transform(images)
print(eigenvalues)

unique_labels = [0,1,2,3,4,5,6,7,8,9]
fig, ax = plt.subplots()
for i in range(len(unique_labels)):
    idxs = np.where(labels == unique_labels[i])
    ax.scatter(t_images[idxs,0], t_images[idxs,1], c=colors[i], label=unique_labels[i],s=10)

ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(unique_labels)):
    idxs = np.where(labels == unique_labels[i])
    ax.scatter(t_images[idxs,0], t_images[idxs,1],t_images[idxs,2], c=colors[i],label=unique_labels[i],s=10)

ax.legend()
plt.show()