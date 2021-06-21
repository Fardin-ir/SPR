import pandas as pd

data = pd.read_csv('data_banknote_authentication.txt', na_values = '?',names=["Variance", "Skewness", "Kurtosis", "Entropy","inauthentic"])
data = data.sample(frac=1)
data.to_csv('s_data.csv',index=False) 