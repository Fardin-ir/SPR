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

def knn_model_selection(k_range,x_train,y_train,x_test,y_test,x_valid,y_valid,distance_method):
    accuracy_train = []
    accuracy_valid = []
    for k in range(1,k_range+1):
        neigh = KNeighborsClassifier(n_neighbors=k, metric=distance_method)
        neigh.fit(x_train, y_train)
        accuracy_train.append(neigh.score(x_train, y_train))
        accuracy_valid.append(neigh.score(x_valid, y_valid))

    df = pd.DataFrame({'k':range(1,21),'Train Error':1-np.asarray(accuracy_train), 'Validation Error':1-np.asarray(accuracy_valid)}).set_index('k')
    print(df)
    max_train=[max(accuracy_train),accuracy_train.index(max(accuracy_train))+1]
    max_valid=[max(accuracy_valid),accuracy_valid.index(max(accuracy_valid))+1]
    plt.plot(range(1,k_range+1),accuracy_train,'r',label=f'train, max is {max_train[0]:.3f} for k={max_train[1]}')
    plt.plot(range(1,k_range+1),accuracy_valid,'b',label=f'valid, max is {max_valid[0]:.3f} for k={max_valid[1]}')
    plt.xticks(range(1,k_range+1))
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(f'distance: {distance_method}')
    plt.show()

#cosine,euclidean
#knn_model_selection(20,x_train,y_train,x_test,y_test,x_valid,y_valid,'cosine')
knn_model_selection(20,x_train,y_train,x_test,y_test,x_valid,y_valid,'euclidean')