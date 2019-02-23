# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
def sigmoid(x):
    '''
    :param x:
    :return:
    '''
    if type(x)!=np.ndarray:
       return 1/(1+math.exp(-x))
    return 1/(1+np.exp(-x))

#激活函数的偏导数
def sigDer(x):
    '''

    :param x:
    :return:
    '''
    return sigmoid(x)*(1-sigmoid(x))

if __name__ == "__main__":
    #N, D_in, H, D_out = 64, 1000, 100, 10
    # Create random input and output data
    xs = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    ys = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    #
    # # Randomly initialize weights
    w1 = np.random.random((2,2))
    w2 = np.random.random((2,2))
    # Create random input and output data

    # xs = np.random.random((N, D_in))
    # ys = np.random.random((N, D_out))
    #
    # # Randomly initialize weights
    # w1 = np.random.random((D_in, H))
    # w2 = np.random.random((H, D_out))

    print(xs)
    print("------------")
    print(ys)
    learning_rate = 0.005
    losses = []
    #learning_rate = 0.05
    for step in range(1000):
       for i in range(len(xs)):
            #计算h层输出
            hin = xs[i].dot(w1)
            #对h层激活
            hout = sigmoid(hin)
            #计算o层输出
            oin = hout.dot(w2)
            #对o层激活
            out = sigmoid(oin)

            y_pred = out

            loss = np.square(y_pred - ys[i]).sum()
            if step%50==0:
               losses.append(loss)
            print(step, loss)

            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_y_pred = 2.0 * (y_pred - ys[i])

            grad_w2 = hout.T.dot(grad_y_pred)
            grad_h_relu = grad_y_pred.dot(w2.T)
            grad_h = grad_h_relu.copy()
            grad_h[hin < 0] = 0
            grad_w1 = xs[i].T.dot(grad_h)

            # Update weights
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2
    plt.plot(losses)
    plt.show()