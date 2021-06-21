import numpy as np
import pandas as pd
import math
import time
import itertools
start_time = time.time()

df = pd.read_csv('s_data.csv')


#number of features
m=2

#returns mode of array 'x'
def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]



def train_test_data(data,n,feature1,feature2):
    train_data = data[[feature1,feature2,'inauthentic']].iloc[:n,:]
    test_data = data[[feature1,feature2,'inauthentic']].iloc[n:,:]
    return train_data,test_data

def euclidean_distance(x,y):
    return np.linalg.norm(x - y)


#sorts data of matrix 'data' based on their distance from point 'x', distance is calculated with 'distance_method'
def get_neighbors_sorted(data,x,distance_method):
    global m
    z = np.zeros((len(data),1), dtype='int64')
    data = np.append(data, z, axis=1)
    for row in range(len(data)):
        data[row,m+1] = distance_method(x,data[row,:m])
    return data[data[:,-1].argsort()] 

#gets train and test set;number of nearest neighbors as 'k' and 'distance method'
#predicts a label for each data of test-set and add it to a new colomn of test-set
#then returns test-set
def knn(train,test,distance_method,k):
    global m
    #convert data_frame to numpy matrix to speed-up calculations
    train = train.copy().values
    test = test.copy().values
    #add new column for predicted data to test-set
    z = np.zeros((len(test),1), dtype='int64')
    test = np.append(test, z, axis=1)
    #predict label for each point in test-set and return it
    for row in range(len(test)):
        x = test[row][:m]
        nearest_neighbors = get_neighbors_sorted(train,x,distance_method)[:k]
        test[row][m+1] = mode1(nearest_neighbors[:,m])
    return test

#returns accuracy
def get_accuracy(actual, predicted):
    return (actual == predicted).sum() / float(len(actual))

#main function




def feature_validation(data,distance_method,k,n):
    combs = list(itertools.combinations(list(df.drop('inauthentic',axis=1).columns.values),2))
    accuracy = []
    for comb in combs:
        train,test = train_test_data(df,n,comb[0],comb[1])
        #call knn for each fold as test-set
        test = knn(train,test,distance_method,k)
        accuracy.append(get_accuracy(test[:,m],test[:,m+1]))
    accuracy = pd.DataFrame(accuracy, columns = ['accuracy'], index=[f'{comb[0]},{comb[1]}' for comb in combs]) 
    return accuracy



print(feature_validation(df,euclidean_distance,3,500))
        
