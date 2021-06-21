import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math 

#import shuffled data
df = pd.read_csv('s_data.csv')[['bill_length_mm','bill_depth_mm','species']].dropna()




#Applys augmentaion and normalization(negetive class*-1) on data and returns training and test set
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



#calculates cost function according to theta 
def cost(X,Y,theta):
    j = np.sum((X.dot(theta)-Y)**2)/(2*X.shape[0])
    return j

#calculates cost function
def get_accuracy(actual, predicted,p_class,n_class):
    temp = []
    for i in range(len(predicted)):
        if predicted[i]>0:
            temp.append(p_class)
        else:
            temp.append(n_class)
    return (actual == temp).sum() / float(len(actual))

#gradient descent
def gradient_descent(X, b, theta, alpha,iterations):
    cost_history = [0] * iterations
    m=X.shape[0]
    #convert pandas dataframe to numpy matrix, to increase computing speed
    X_np = X.values
    #main loop, using matrix operations 
    for iteration in range(iterations):
        h = X_np.dot(theta)
        gradient = X_np.T.dot(h-b)*1/(m)
        theta = theta - alpha*gradient
        cost_history[iteration] = cost(X,b,theta)
    return theta, cost_history

#newton
def newton(X, b, theta,alpha,iterations):
    cost_history = [0] * iterations
    m=X.shape[0]
    X_np = X.values
    for iteration in range(iterations):
        h = X_np.dot(theta)
        d1 = X_np.T.dot(h-b)*1/(m)
        d2 = X_np.T.dot(X_np)*(1/m)
        theta = theta - np.linalg.inv(d2).dot(d1)
        print(theta)
        cost_history[iteration] = cost(X,b,theta)
    return theta, cost_history

#main function of this program
def MSE(alpha,iterations,data,p_class,n_class,method):

    train_data, test_data = prepare_data(data,300,p_class,n_class)

    X_train =  train_data.drop('species',axis=1)
    Y_train = train_data['species']

    X_test =  test_data.drop('species',axis=1)
    Y_test = test_data['species']

    theta = np.zeros(X_train.shape[1])
    newtheta, cost_history = method(X_train, np.ones(len(X_train)), theta, alpha,iterations)
    print('theta =',newtheta)

    predicted_test = X_test.dot(newtheta)

    accuracy = get_accuracy(Y_test, predicted_test.values,p_class,n_class)
    print('accuracy: ',accuracy)
    
    p_class_data = data.loc[data['species'].isin([p_class])].drop('species',axis=1).values
    n_class_data = data.loc[data['species'].isin([n_class])].drop('species',axis=1).values
    #plot
    plt.plot(p_class_data[:,0],p_class_data[:,1], 'ro',markersize=2, label=p_class)
    plt.plot(n_class_data[:,0],n_class_data[:,1], 'bo',markersize=2, label=n_class)
    x = np.linspace(30., 60.)
    y = np.linspace(10., 22.)[:, None]
    plt.contour(x, y.ravel(), newtheta[1]*x + newtheta[2]*y + newtheta[0], [0])
    plt.legend()
    plt.title(f'{newtheta[1]:.3f}x{newtheta[2]:.3f}y{ newtheta[0]:.3f}=0, {method.__name__},accuracy={accuracy:.3f},iteration={len(cost_history)}')
    plt.show()

    plt.plot(range(1,len(cost_history)+1),cost_history)
    plt.xlabel('iteration')
    plt.ylabel('MSE cost')
    plt.title(method.__name__)
    plt.show()

#call main function
MSE(0.0009,500,df,'Gentoo','Adelie',gradient_descent)
MSE(0.0009,500,df,'Gentoo','Adelie',newton)