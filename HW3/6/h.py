import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math 

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

#gradient descent
def gradient_descent(X, b, theta, alpha,iterations):
    m=X.shape[0]
    X_np = X.values
    pocket_i = []
    pocket = []
    m_sampels = 9999
    for iteration in range(iterations):
        h = X_np.dot(theta)
        theta_old = theta
        gradient = np.sum(X_np[np.where(h <= 0)],axis=0).T
        theta = theta + alpha*gradient
        if len(X_np[np.where(h <= 0)]) < m_sampels:
            m_sampels = len(X_np[np.where(h <= 0)])
            pocket_i.append(iteration)
            pocket = theta
        print(theta)
        if (theta_old == theta).all():
            break
    return pocket,iteration+1,pocket_i


#main function of this program
def MSE(alpha,iterations,data,p_class,n_class):

    train_data, test_data = prepare_data(data,300,p_class,n_class)

    X_train =  train_data.drop('species',axis=1)
    Y_train = train_data['species']

    X_test =  test_data.drop('species',axis=1)
    Y_test = test_data['species']

    theta = np.zeros(X_train.shape[1])
    newtheta,iteration,pocket_i = gradient_descent(X_train, np.ones(len(X_train)), theta, alpha,iterations)
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
    plt.title(f'{newtheta[1]:.3f}x{newtheta[2]:.3f}y{ newtheta[0]:.3f}=0, accuracy={accuracy:.3f}, iteration={iteration}')
    print(f'pocket updated {len(pocket_i)} times and best answer was in {pocket_i[-1]+1}th iteration')
    plt.show()

#call main function
MSE(0.0009,2000,df,'Adelie','Gentoo')
