import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

p1 = pd.read_csv('p1.csv', header=None)
p2 = pd.read_csv('p2.csv', header=None)

y_true = np.append(np.ones(1000),np.zeros(1000))

print('1:',f1_score(y_true,p1.values))
print('2:',f1_score(y_true,p2.values))
