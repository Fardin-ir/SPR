from scipy.special import comb
from matplotlib import pyplot as plt
import numpy as np

k=25
m=200
omega=10000
n = np.linspace(200,300,300)

g=(1/(comb(m-1,k-1))-1/comb(omega,k-1))
f=1/(comb(n,k)*k/(k-1)*g)

plt.plot(n,f, 'r')
plt.show()
