#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time as t
import matplotlib.pyplot as plt

X_book = np.random.rand(100,1)
Y_book = 4 + 3*X_book + np.random.randn(100,1)
X_book_b = np.c_[np.ones((100,1)),X_book]

df = pd.read_csv("data.txt")

m = len(df['Y'])
X = df['X']
X_b = np.c_[np.ones((m,1)),X]
Y = np.array(df['Y'])
Y.resize(m,1)

plt.scatter(X,Y)

#Using the normal equation for computing the best value for theta
def normal_equation():
    t1 = t.time()
    theta_normal =  np.linalg.inv( X_b.T.dot(X_b )) .dot(X_b.T).dot(Y)
    t2 = t.time()
    x_plot1 = np.linspace(df['X'].min(),df['Y'].max())
    y_plot1 = theta_normal[0] + theta_normal[1]*x_plot1
    plt.plot(x_plot1,y_plot1,"r-",label="Normal Equation")
    return theta_normal,format(t2-t1,"f")

#Batch-Gradient Descent
def batch_gradient_descent(eta,epochs):
    theta = np.random.randn(2,1)
    t1 = t.time()
    for epoch in range(epochs):
        gradient = (2/m) * X_b.T.dot(X_b.dot(theta) - Y)
        theta-= eta*gradient
    t2 = t.time()
    x_plot2 = np.linspace(df['X'].min(),df['Y'].max())
    y_plot2 = theta[0] + theta[1]*x_plot2
    plt.plot(x_plot2,y_plot2,"b-")
    return theta,format(t2-t1,"f")


#Stochastic Gradient Descent (t0 and t1 are learning schedule hyper params)
def stoch_gradient_descent(eta,epochs):
    theta = np.random.randn(2,1)
    t1 = t.time()
    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = Y[random_index:random_index+1]
            gradient = 2*xi.T.dot(xi.dot(theta) -yi)
            #eta = learning_schedule(epoch*10 +i)
            theta = theta - eta*gradient
    t2 = t.time()
    return theta,format(t2-t1,"f")


#Mini-Batch Gradient Descent (t0 and t1 are learning schedule hyper params)
def mini_batch_gradient_descent(eta,batch_size,epochs):
    theta = np.random.randn(2,1)
    t1 = t.time()
    for epoch in range(epochs):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+batch_size]
        yi = Y[random_index:random_index+batch_size]
        gradient =(2/batch_size)*xi.T.dot(xi.dot(theta) -yi)
        #eta = learning_schedule(epoch*10 +i)
        eta = 0.01
        theta = theta - eta*gradient
    t2 = t.time()
    return theta,format(t2-t1,"f")




if __name__ == "__main__":
    theta_normal,time_normal = normal_equation()
    theta_batch,time_batch = batch_gradient_descent(0.01,1000)
    theta_stoch,time_stoch = stoch_gradient_descent(0.001,1000)
    theta_mini_batch,time_mini_batch = mini_batch_gradient_descent(0.01,50,1000)
    print("\n Theta from Normal Equation: {0},{1} \n Time taken: {2} seconds".format(theta_normal[0],theta_normal[1],time_normal))
    print("----------------------------")
    print("\n Theta from Batch-Gradient Descent: {0},{1} \n Time taken: {2} seconds".format(theta_batch[0],theta_batch[1],time_batch))
    print("----------------------------")
    print("\n Theta from Stochastic-Gradient Descent: {0},{1} \n Time taken: {2} seconds".format(theta_stoch[0],theta_stoch[1],time_stoch))
    print("----------------------------")
    print("\n Theta from Mini-Batch-Gradient Descent: {0},{1} \n Time taken: {2} seconds".format(theta_mini_batch[0],theta_mini_batch[1],time_mini_batch))
    #plt.show()
