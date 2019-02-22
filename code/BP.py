import numpy as np
import math
import matplotlib.pyplot as plt
#定义激活函数,可以计算的那个元素以及数组
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

#实现向量与矩阵相乘
def dot(w,x):
    '''

    :param w:
    :param x:
    :return:
    '''
    #先将行向量转换为列向量
    x = np.transpose(x)
    #计算相乘
    return np.transpose(np.dot(w,x))

#将计算获取的向量进行激活
def sigmoidW(x):
    '''
    :param w:
    :return:
    '''
    return sigmoid(x)

#计算输出层的实际值与深数据真实值的差
def minus(Y_,Y):
    '''
    :param Y_: 实际值
    :param Y: 期望值
    :return:
    '''
    return (Y_ - Y)
#计算实际值与期望值之间的误差
def loss(Y_,Y):
    '''
        :param Y_: 实际值
        :param Y: 期望值
        :return:
    '''
    return np.power(minus(Y_,Y),2)/2
#计算w2的偏导数，并返回
def wODer(Y_,Y,out,hout):

    w = np.ones((2,2))
    return w*minus(Y_,Y)*sigDer(out)*sigmoid(hout)

#实现矩阵更新
def updateW(w,w_,k):
    '''
    :param w: w是原矩阵
    :param w_: w_是偏导数矩阵
    :param k: k是更新幅度，也是学习速率
    :return:
    '''
    return (w-k*w_)

def wHDer(Y_,Y,out,w2,hout,X):

    w = np.ones((2,2))

    for i in range(len(w)):
        for j in range(len(w[i])):
            t = float(0.0)
            for n in range(len(Y_)):
                t = t+(Y_[n]-Y[n])*sigDer(out[n])*w2[n][i]*X[j]

            w[i][j]=t
    return w


if __name__ == "__main__":

    #定义行向量,仅有一个数据记录
    xs = np.array([[1.0,1.0],[1.0,0.0],[0.0,1.0],[0.0,0.0]])
    ys = np.array([[1.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]])

    # 初始化i层到h层的权重
    w1 = np.random.random((2, 2))
    # 初始化h层到o层的权重
    w2 = np.random.random((2, 2))

    losses = []

    for step in range(100000):
        for i in range(len(xs)):

            X = xs[i]
            Y = ys[i]
            # print("输入向量")
            # print(X)

            #计算h层输出
            hin = dot(w1,X)
            # print("h层计算输出")
            # print(hin)

            #将输出的h层激活
            hout = sigmoidW(hin)
            # print("h激活输出")
            # print(hout)

            #将h层数据传到o层，并计算输出
            oin = dot(w2,hout)
            # print("o层输出")
            # print(oin)

            #将输出的o层激活
            out = sigmoidW(oin)
            Y_ = out
            # print("o层激活输出")
            # print(out)

            print("损失")
            print(round( np.sum(loss(out,Y)),6))
            if step%5000==0:
               losses.append(round( np.sum(loss(out,Y)),6))
            #获取最终结果后，对w2层求偏导数,w2的偏导数矩阵是w_2
            w_2 = wODer(Y_,Y,out,hout)
            # print("h-o的偏导数")
            #更新w2
            w2 = updateW(w2,w_2,0.05)
            # print("更新后的w2")
            # print(w2)

            #开始对h层的w1进行求偏导数
            w_1 = wHDer(Y_,Y,out,w2,hout,X)
            # print("i-h的偏导数")
            # print(w_1)

            w1 = updateW(w1,w_1,0.05)
            # print("更新后的w1")
            # print(w1)

    print(losses)
    plt.plot(losses)
    plt.show()