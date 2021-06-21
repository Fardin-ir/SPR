import numpy as np
import matplotlib.pyplot as plt

mu1 = np.array([-3,0])
mu2 = np.array([3,0])
cov1 = np.array([[1.5,1],[1,1.5]])
cov2 = np.array([[1.5,-1],[-1,1.5]])

x1, y1 = np.random.multivariate_normal(mu1, cov1, 500).T
x2, y2 = np.random.multivariate_normal(mu2, cov2, 500).T

plt.plot(x1, y1, 'ro',label='class 1',markersize=1)
plt.plot(x2, y2, 'bo',label='class 2',markersize=1)
plt.legend()
plt.axvline(x=-0.451)
plt.axis('equal')
plt.show()