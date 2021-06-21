import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('1KGP.txt', delimiter=' ',header=None)
Y = data.iloc[:,3:]
labels = data.iloc[:,2].unique()
colors = ['b','g','r','c','m','y','k']
modes = Y.mode().iloc[0]
Y = (Y!=modes).astype(int)
Y = Y-Y.mean()

t_data = PCA(n_components=2).fit(Y).transform(Y)

fig, ax = plt.subplots()
for i in range(len(labels)):
    idxs = np.where(data.iloc[:,2] == labels[i])
    ax.scatter(t_data[idxs,0], t_data[idxs,1], c=colors[i], label=labels[i])

ax.legend()
plt.show()
