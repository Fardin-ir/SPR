import pandas as pd

data = pd.read_csv('penguins.csv', na_values = '?')
data = data.sample(frac=1)
data.to_csv('s_data.csv',index=False) 