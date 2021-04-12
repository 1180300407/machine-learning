# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class logistic_function:
    def __init__(self, train_N=150, test_N=150, dim=2, eta=0.1, lamda=0.03):
        '''
        构造函数
        train_N:训练数据的规模
        valid_N:验证数据的规模
        test_N :测试数据的规模
        order  :模拟的多项式阶数
        start  :生成数据的起点
        end    :生成数据的终点
        eta    :学习率
        '''
        #根据给定参数随机生成数据
        self.dim = dim
        self.eta = eta
        self.lamda = lamda
        self.train_N = train_N
        self.test_N = test_N
        all_data = self.get_data(train_N, test_N)
        self.train_data = all_data[0]
        self.train_normaldata = all_data[1]
        self.train_label = all_data[2]
        self.test_data = all_data[3]
        self.test_label = all_data[4]
    
    #生成数据，为了便于展示，这里选择生成二维数据
    #train_N:训练数据规模
    #test_N :测试数据规模
    def get_data(self, train_N, test_N):
        #设定随机数种子
        np.random.seed(20711018)

        #手工设定模型的真实参数值
        theta_0 = 1
        theta_1 = 2
        theta_2 = 5

        #设定特征取值范围，为[-x_width,x_width]
        x_width = 6

        #每一维特征上的样本值(N*1列向量)，用均匀分布生成
        #第0维为全1
        x0 = np.ones((train_N,1))

        #第一维取值范围为[-x_width,x_width]
        x1 = np.random.rand(train_N, 1)*x_width*2-x_width

        #改变随机种子，使得两个维度的特征独立同分布
        np.random.seed(213715360)
        #第二维取值范围为[-x_width,x_width]
        x2 = np.random.rand(train_N, 1)*x_width*2-x_width
        #测试不满足朴素贝叶斯情况的取值
        #x2=x1
        X = np.hstack((x0,x1,x2))
        min_x = x_width
        max_x = -x_width
        for i in range(train_N):
            for j in range(self.dim):
                if(X[i][j+1]<min_x):
                    min_x = X[i][j+1]
                if(X[i][j+1]>max_x):
                    max_x = X[i][j+1]
        coef = 1.0/(max_x-min_x)
        X_coef = coef*(X-min_x)

        #利用设定的真实参数值代入sigmoid函数，并加入噪声，得到标签
        t = self.sigmoid_function(theta_0*x0+theta_1*x1+theta_2*x2)+np.random.randn(train_N,1)*0.12
        t = np.round(t)


        #测试数据集
        #重新设定随机数种子
        np.random.seed(2720101960)
        #每一维特征上的样本值(N*1列向量)，用均匀分布生成
        #第0维为全1
        x0_test = np.ones((test_N,1))
    
        #第一维取值范围为[-x_width,x_width]
        x1_test = np.random.rand(test_N,1)*x_width*2-x_width

        #改变随机种子，使得两个维度的特征独立同分布
        np.random.seed(2044460)
        #第二维取值范围为[-x_width,x_width]
        x2_test = np.random.rand(test_N,1)*x_width*2-x_width
    
        X_test = np.hstack((x0_test,x1_test,x2_test))

        #利用设定的真实参数值代入sigmoid函数，得到标签
        t_test = self.sigmoid_function(theta_0*x0_test+theta_1*x1_test+theta_2*x2_test)
        t_test = np.round(t_test)
        
        #样本真实情况
        plt.title('true labels' )
        self.draw(X, t, train_N)
        plt.plot(x1, x1*(-1)*theta_1/theta_2-theta_0/theta_2,'r')
        plt.ylim(-5,5)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
        
        return X, X_coef, t, X_test, t_test

    #sigmoid函数
    def sigmoid_function(self, z):
        func = 1.0+np.exp((-1)*z)
        return 1.0/func


    #预测函数
    #theta：模型参数，m*1列向量
    #a    ：样本数据，N*m矩阵
    def hypothesis_function(self, theta,a):        
        mat = np.dot(a,theta)
        value = self.sigmoid_function(mat)
        return value

    #损失函数
    #theta   ：模型参数，m*1列向量
    #a       ：样本数据，N*m矩阵
    #c       : 样本标签，N*1列向量
    #Datasize: 样本大小，N
    #J(theta)=[(-1)/Datasize]*[t^T*log(h(x))+(1-t^T)log(1-h(x))]
    def loss_function(self, a,theta,c,Datasize):
        value = 0
        h_0 = np.log(1-self.hypothesis_function(theta,a))
        h_1 = np.log(self.hypothesis_function(theta,a))
        value = np.dot(c.T,h_1)+np.dot(1-c.T,h_0)
        value = value*(-1)
        return value/Datasize

    #加惩罚项的损失函数
    #theta   ：模型参数，m*1列向量
    #a       ：样本数据，N*m矩阵
    #c       : 样本标签，N*1列向量
    #Datasize: 样本大小，N
    #J(theta)=[(-1)/Datasize]*[t^T*log(h(x))+(1-t^T)log(1-h(x))]+0.5*lamda||theta||^2/Datasize
    def loss_function_reg(self,a,theta,c,Datasize,lam):
        value = self.loss_function(a,theta,c,Datasize)
        reg = 0.5*lam*theta.T@theta/Datasize
        return value+reg

    #梯度函数
    #theta   ：模型参数，m*1列向量
    #a       ：样本数据，N*m矩阵
    #c       : 样本标签，N*1列向量
    #Datasize: 样本大小，N
    def gradient_function(self,theta,a,c,Datasize):
        #grad=X^T(h(X)-t)/Datasize
        value = np.array(self.hypothesis_function(theta,a))
        value = value.reshape(-1,1)
        grad = value-c
        grad = np.dot(a.T,grad)
        grad = grad/Datasize
        return grad

    #将两类点分别画出来
    #x   : 样本数据
    #y   : 样本标签
    #size: 样本大小
    def draw(self, x,y,size):
        c = ['b','g']
        x1_b = []
        x1_g = []
        x2_b = []
        x2_g = []
        for i in range(size):
            if(y[i]==0):
                x1_b.append(x[i][1])
                x2_b.append(x[i][2])
            else:
                x1_g.append(x[i][1])
                x2_g.append(x[i][2])
        plt.scatter(x1_b,x2_b,c=c[0])
        plt.scatter(x1_g,x2_g,c=c[1])
    
    #计算准确率
    def precision(self, y,t,size):
        count = 0
        for i in range(size):
            if(y[i]==t[i]):
                count += 1
            
        return (float(count))/size
    
    #不带正则项的logistic_regression分类
    def classify_noregular(self):
        alpha = self.eta
        x1 = self.train_data[:,1]
        x1_test = self.test_data[:,1]
        
        #无惩罚项
        theta_nreg = np.array([0]*(self.dim+1))
        theta_nreg = theta_nreg.T
        theta_nreg = theta_nreg.reshape(-1,1)

        loss0_nreg = 0
        loss1_nreg = self.loss_function(self.train_normaldata, theta_nreg, self.train_label, self.train_N)
        terminal = 1e-7

        while(abs(loss1_nreg-loss0_nreg)>terminal):
            theta_temp = theta_nreg-alpha*self.gradient_function(theta_nreg, self.train_normaldata, self.train_label, self.train_N)
            loss0_nreg = loss1_nreg
            loss1_nreg = self.loss_function(self.train_normaldata, theta_temp, self.train_label, self.train_N)
            if(loss1_nreg>loss0_nreg):
                alpha = alpha/2
                continue
    
            theta_nreg=theta_temp
    

        #模型学习情况
        plt.title('learning model without regular items' )
        self.draw(self.train_data, self.train_label, self.train_N)
        plt.plot(x1,x1*(-1)*theta_nreg[1]/theta_nreg[2]-theta_nreg[0]/theta_nreg[2],'r')
        plt.ylim(-5,5)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

        #用测试集测试
        plt.title('test without regular items' )
        self.draw(self.test_data, self.test_label, self.test_N)
        plt.plot(x1_test, x1_test*(-1)*theta_nreg[1]/theta_nreg[2]-theta_nreg[0]/theta_nreg[2], 'r')
        plt.ylim(-5,5)
        plt.xlabel('x1_test')
        plt.ylabel('x2_test')
        plt.show()
        y_nreg = self.hypothesis_function(theta_nreg, self.test_data)
        y_nreg = np.round(y_nreg)
        acc = self.precision(y_nreg, self.test_label, self.test_N)
        print('Precision without regular items: %f%%' %(100*acc))
    
    #带正则项的logistic_regression分类
    def classify_withregular(self):
        alpha = self.eta
        x1 = self.train_data[:,1]
        x1_test = self.test_data[:,1]
        #带惩罚项
        theta_reg = np.array([0]*(self.dim+1))
        theta_reg = theta_reg.T
        theta_reg = theta_reg.reshape(-1,1)

        loss0_reg = 0
        loss1_reg = self.loss_function_reg(self.train_normaldata, theta_reg, self.train_label, self.train_N, self.lamda)
        terminal = 1e-10

        while(abs(loss1_reg-loss0_reg)>terminal):
            theta_temp = theta_reg*(1-self.eta*self.lamda/self.train_N)-self.eta*self.gradient_function(theta_reg, self.train_normaldata, self.train_label, self.train_N)
            loss0_reg = loss1_reg
            loss1_reg = self.loss_function_reg(self.train_normaldata, theta_temp, self.train_label, self.train_N, self.lamda)
            if(loss1_reg>loss0_reg):
                alpha = alpha/2
                continue
    
            theta_reg=theta_temp
        
        #模型学习情况
        plt.title('learning model with regular items,lamda=%f' %self.lamda)
        self.draw(self.train_data, self.train_label, self.train_N)
        plt.plot(x1, x1*(-1)*theta_reg[1]/theta_reg[2]-theta_reg[0]/theta_reg[2],'r')
        plt.ylim(-5,5)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
    
        #用测试集测试
        plt.title('test with regular items,lamda=%f' %self.lamda)
        self.draw(self.test_data, self.test_label, self.test_N)
        plt.plot(x1_test,x1_test*(-1)*theta_reg[1]/theta_reg[2]-theta_reg[0]/theta_reg[2],'r')
        plt.ylim(-5,5)
        plt.xlabel('x1_test')
        plt.ylabel('x2_test')
        plt.show()
        y_reg = self.hypothesis_function(theta_reg, self.test_data)
        y_reg = np.round(y_reg)
        acc = self.precision(y_reg, self.test_label, self.test_N)
        print('Precision with regular items: %f%%' %(100*acc))
        
def main():
    lc = logistic_function()
    lc.classify_noregular()
    lc.classify_withregular()
    
if __name__ == '__main__':
    main()