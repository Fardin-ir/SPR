import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('1KGP.txt', delimiter=' ',header=None)
Y = data.iloc[:,3:]
labels = data.iloc[:,1].unique()
colors = ['b','g','r','c','m','y','k']
modes = Y.mode().iloc[0]
Y = (Y!=modes).astype(int)

pca = PCA(n_components=3).fit(Y)

components = pca.components_[2,:]
plt.scatter(range(len(components)),np.abs(components))
plt.show()