import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random 

with open('t10k-labels-idx1-ubyte','rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                            offset=8)
    
with open('t10k-images-idx3-ubyte','rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(labels), 784)

colors = ['b','g','r','c','m','y','k','tab:gray','lime','orange']


pcs = PCA(n_components=185).fit(images)

print(f'nuumber of principal components={185}', 'explained variance ratio=',np.sum(pcs.explained_variance_ratio_))

t_images = pcs.transform(images)
re_images = pcs.inverse_transform(t_images)

fig, axs = plt.subplots(3, 2)
fig.tight_layout()
for i in range(3):
    x = random.randint(0,10000)
    axs[i,0].imshow(images[x,:].reshape(28,28),cmap='gray')
    axs[i,0].set_title('Original',size=10)
    axs[i,1].imshow(re_images[x,:].reshape(28,28),cmap='gray')
    axs[i,1].set_title('Reconstructed',size=10)
plt.show()
