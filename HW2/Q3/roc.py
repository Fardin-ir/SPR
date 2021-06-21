import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

#1 for stigma and 0 for other
true_lables =[0,1,0,1,0,0,0,0]
fs1 = pd.read_csv('fs1.csv').values
fs2 = pd.read_csv('fs2.csv').values
fs3 = pd.read_csv('fs3.csv').values


score1=fs1[:,0]-fs1[:,1]
fpr1,tpr1,th=roc_curve(true_lables,score1)
plt.plot(fpr1,tpr1,label='fs1')


score2=fs2[:,0]-fs2[:,1]
fpr2,tpr2,th=roc_curve(true_lables,score2)
plt.plot(fpr2,tpr2,label='fs2')


score3=fs3[:,0]-fs3[:,1]
fpr3,tpr3,th=roc_curve(true_lables,score3)
plt.plot(fpr3,tpr3,label='fs3')
plt.legend()
plt.show()
