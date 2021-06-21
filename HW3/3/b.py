import numpy as np
import pandas as pd
import math
import time
import itertools
import matplotlib.pyplot as plt
start_time = time.time()

df = pd.read_csv('s_data.csv')


#number of features
m=2

#returns mode of array 'x'
def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]

#gets number of folds and returns 'i'th fold as test-set and remaining data as train-fold

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
def knn(train,test,distance_method,k,h):
    global m
    #convert data_frame to numpy matrix to speed-up calculations
    train = train.copy().values
    #add new column for predicted data to test-set
    z = np.zeros((h,1), dtype='int64')
    ##test = np.append(test, z, axis=1)
    #predict label for each point in test-set and return it
    pred = []
    for i in range(h):
        for j in range(h):
            x = [test[0,i,0],test[0,j,1]]
            nearest_neighbors = get_neighbors_sorted(train,x,distance_method)[:k]
            pred.append(mode1(nearest_neighbors[:,m]))
    colors=[]
    for i in pred:
        if int(i) == 1:
            colors.append(0)
        else:
            colors.append(255)
    return np.asarray(colors).reshape(h,h)

#returns accuracy
def get_accuracy(actual, predicted):
    return (actual == predicted).sum() / float(len(actual))

#main function
#gets data,number of folds for k-fold cross validation, distance method and number of nearest neighbors
#return sum of confusion matrixes and mean of accuracy


def diagram(data,distance_method,k,n,featuures,h):
    train,test = train_test_data(df,n,featuures[0],featuures[1])
    test_x = np.linspace(start=min(data[featuures[0]]), stop=max(data[featuures[0]]), num=h)
    test_y = np.linspace(start=min(data[featuures[1]]), stop=max(data[featuures[1]]), num=h)
    test = np.dstack((test_x,test_y))
    colors = knn(train,test,distance_method,k,h)
    fig, ax = plt.subplots()
    ax.pcolormesh(test_x,test_y, colors,cmap='gray')
    p_class = data.loc[data['inauthentic'].isin([1])][[featuures[0],featuures[1]]].values
    n_class = data.loc[data['inauthentic'].isin([0])][[featuures[0],featuures[1]]].values
    plt.legend()
    plt.xlabel(featuures[0])
    plt.ylabel(featuures[1])
    plt.title(f'k={k}')
    plt.show()


#call main function
#cross_validation(data,num_fold,distance_method,k)
print(diagram(df,euclidean_distance,3,500,['Variance','Skewness'],300))
        