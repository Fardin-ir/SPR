import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math 
import time


#import shuffled data
df = pd.read_csv('s_data.csv')[['bill_length_mm','bill_depth_mm','species']].dropna()


    
def prepare_data(data,n,p_class,n_class):
    data = data.copy()
    data.insert(0,'y0',[1]*data.shape[0])
    train_data = data.iloc[:n,:]
    test_data = data.iloc[n:,:]
    train_data = train_data.loc[train_data['species'].isin([p_class,n_class])]
    test_data = test_data.loc[test_data['species'].isin([p_class,n_class])]

    for col in train_data.drop('species',axis=1):
        train_data[col] = np.where(train_data['species'] == n_class, train_data[col].apply(lambda x:-x), train_data[col])
    return train_data,test_data

#print(prepare_data(df,300,'Chinstrap','Gentoo'))


#calculate error function according to theta 
def cost(X,Y,theta):
    j = np.sum((X.dot(theta)-Y)**2)/(2*X.shape[0])
    return j

def get_accuracy(actual, predicted,p_class,n_class):
    temp = []
    for i in range(len(predicted)):
        if predicted[i]>0:
            temp.append(p_class)
        else:
            temp.append(n_class)
    return (actual == temp).sum() / float(len(actual))

def gradient_descent(X, b, theta, alpha):
    start_time = time.time()
    cost_history = [0] * 10000
    m=X.shape[0]
    X_np = X.values
    t=0
    print(alpha)
    for iteration in range(10000):
        h = X_np.dot(theta)
        gradient = X_np.T.dot(h-b)*1/(m)
        old_tetha=theta
        theta = theta - alpha*gradient
        cost_history[iteration] = cost(X,b,theta)
        if (np.absolute(theta-old_tetha) < [0.000001]*3).all():
            t=time.time()-start_time
            break
    return theta, cost_history,t

#main function of this program
def MSE(data,p_class,n_class,method):

    train_data, test_data = prepare_data(data,300,p_class,n_class)

    X_train =  train_data.drop('species',axis=1)
    Y_train = train_data['species']

    X_test =  test_data.drop('species',axis=1)
    Y_test = test_data['species']

    arr = np.linspace(0.00005,0.001,num=50)
    time = []
    for i in arr:    
        theta = np.zeros(X_train.shape[1])
        newtheta, cost_history,t = method(X_train, np.ones(len(X_train)), theta, i)
        if t == 0:
            print('maximuum learning rate: ',i)
            break
        time.append(t)
    time = time[:-1]

    plt.plot(arr[:len(time)],time)
    plt.xlabel('alpha')
    plt.ylabel('time(sec)')
    plt.show()

#call main function
MSE(df,'Gentoo','Adelie',gradient_descent)
