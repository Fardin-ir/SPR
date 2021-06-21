import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('ALS_train.csv').drop(['ID','SubjectID'],axis=1)
data=(data-data.mean())/data.std()
cov_mat = data.cov().abs()['ALSFRS_slope']

print(cov_mat.sort_values(ascending=False).head(11))