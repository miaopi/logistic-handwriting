from numpy import *
import numpy as np
import random
import math

class LogisticRegression(object):

    def __init__(self):
        self.w=None
        self.W=None
        
    def sigmoid(self,X):
        return longfloat(1.0 / (1.0 + np.exp(-X)))

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N。 X_batch：一个形状（n，d）的numpy数组，其中包含一个n的小批次
        data points; each point has dimension D. 数据点；每个点都有维度D
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.。y_batch：包含小批次标签的形状（n，）的numpy数组。

        Returns: A tuple containing:
        - loss as a single float 单浮损失
        - gradient with respect to self.W; an array of the same shape as W 相对于self.w的梯度；
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        m=X_batch.shape[0]        
        h=self.sigmoid(X_batch.dot(self.w))               
        loss=(-y_batch.T.dot(np.log(h))-(1-y_batch).T.dot(np.log(1-h)))/m   
        grad=X_batch.T.dot(h - y_batch)/m        
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        return loss,grad
    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.使用随机梯度下降训练这个线性分类器。
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N 包含训练数据的形状（n，d）的numpy数组；有n个
         training samples each of dimension D.                       每个维D的训练样本。
        - y: A numpy array of shape (N,) containing training labels; 一个包含训练标签的形状（n，）的numpy数组；
        - learning_rate: (float) learning rate for optimization. 学习率：（浮动）优化学习率
        - num_iters: (integer) number of steps to take when optimizing （integer）优化时要采取的步骤数
        - batch_size: (integer) number of training examples to use at each step.批处理大小：（整数）每个步骤要使用的培训示例数。
        - verbose: (boolean) If true, print progress during optimization.如果为真，则在优化期间打印进度。

        Outputs:
        A list containing the value of the loss function at each training iteration.包含每次训练迭代时损失函数值的列表。
        """
        num_train, dim = X.shape
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            Sample_batch = np.random.choice(num_train, batch_size)
            X_batch = X[Sample_batch]
            y_batch = y[Sample_batch]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w -=learning_rate*grad#更新权重
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[0])#生成一个空数组 0是行，1是列
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        m=X.shape[0]   
        #X = np.hstack((X, np.mat(np.ones((m,1)))))   
        #y_pred_list=[]   
        #for xx in X:            
         #   y_pred = self.sigmoid(xx.dot(self.w))                 
          #  if y_pred>0.5:                
           #     y_pred_list.append(1)            
            #else:                
             #   y_pred_list.append(0)  
        #return y_pred_list
        y_pred=self.sigmoid(X.dot(self.w))
        for i in range(m):
            if y_pred[i]>0.5:
                y_pred[i]=1
            else:
                y_pred[i]=0
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose = True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """
        num_train, dim = X.shape#num_train是行数 14000，dim是列数 785
        self.W = np.zeros((dim,10))#生成一个785*10的空数组
        for it in range(10):
            y_train = []
            for label in y:
                if label == it:
                    y_train.append(1)#后面加上
                else:
                    y_train.append(0)
            y_train = np.array(y_train)#生成数组
            self.w = None
            print("it = ", it)
            self.train(X,y_train,learning_rate, num_iters ,batch_size)
            self.W[:,it] = self.w#切片，然后将更新后的权值放入相应的类别里面
            print(self.W.shape)
    def one_vs_all_predict(self,X):
        laybels = self.sigmoid(X.dot(self.W))
        y_pred = np.argmax(laybels,axis=1)#返回最大索引值，索引值为0-9
        print(laybels.shape)
        return y_pred
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            