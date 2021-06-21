import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import math

clean_fake = pd.read_csv('clean_fake.txt',names=['header'])
clean_real = pd.read_csv('clean_real.txt',names=['header'])


def euclidean_distance(x,y):
    return np.linalg.norm(x - y)


def cosine_distance(x,y):
    return 1-np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def split_data(x,y,train_frac,test_frac):
    i = math.floor(train_frac*x.shape[0])
    j = math.floor(test_frac*y.shape[0])
    x_train = x[:i,:]
    y_train = y[:i]
    x_test = x[i:i+j,:]
    y_test = y[i:i+j]
    x_valid = x[i+j:,:]
    y_valid = y[i+j:]
    return x_train,y_train,x_test,y_test,x_valid,y_valid

def prepare_data(clean_fake,clean_real):
    clean_fake.insert(1,'output',np.ones(len(clean_fake)))
    clean_real.insert(1,'output',np.zeros(len(clean_real)))
    dataset = pd.concat([clean_fake,clean_real],axis=0, ignore_index=True)
    dataset = dataset.sample(frac=1)
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(dataset['header']).toarray()
    y = dataset['output'].values
    return split_data(x,y,0.7,0.15)
    

x_train,y_train,x_test,y_test,x_valid,y_valid = prepare_data(clean_fake,clean_real)
