# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

class conjugate_gradient:
    '''
    最小二乘法的共轭梯度解法，这里数据由[start,end]间的高斯函数随机生成
    '''
    def __init__(self, train_N=50, valid_N=50, test_N=50, order=6, start=0, end=1, eta=0.1):
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
        all_data = self.get_data(train_N, valid_N, test_N, order, start, end)
        self.train_data = all_data[0]
        self.train_x = all_data[1]
        self.train_label = all_data[2]
        self.valid_data = all_data[3]
        self.valid_x = all_data[4]
        self.valid_label = all_data[5]
        self.test_data = all_data[6]
        self.test_x = all_data[7]
        self.test_label = all_data[8]
        self.order = order
        self.eta = eta
        '''
        print('train_data:')
        print(self.train_data)
        print('valid_data:')
        print(self.valid_data)
        print('test_data:')
        print(self.test_data)
        '''
    
    def get_data(self, train_N, valid_N, test_N, order, start, end):
        '''
        随机生成数据
        train_N:训练数据的规模
        valid_N:验证数据的规模
        test_N :测试数据的规模
        order  :模拟的多项式阶数
        start  :生成数据的起点
        end    :生成数据的终点
        '''        
        
        #pi值
        pi = np.pi

        #添加的高斯噪声的均值与方差
        mu = 0
        sigma = 0.12
        X = np.ones((train_N, order+1))
    
        #生成x矩阵
        for i in range(train_N):
            for j in range(order+1):
                X[i][j] = np.power(start + i*(end-start)/train_N, j)

        #存储真实值列向量
        t = []
        #存储所取到的x值
        x = []

        #真实函数值&添加噪声
        for i in range(train_N):
            x.append(X[i][1])
            f_x = np.sin(2*pi*X[i][1])+random.gauss(mu, sigma)  #在for循环中根据x值生成正弦函数值
            t.append(f_x)  #加入到真实值列表中
    
        #转为列向量
        t = np.array(t) 
        t = t.reshape(-1, 1)
    
        #验证数据集，用来确定超参数lamda
        validation_x = []
        validation_X = np.ones((valid_N, order+1))
        t_validation = []
        for i in range(valid_N):
            ran_num = random.randrange(0,100*valid_N)/(100*valid_N)
            while(ran_num in x):
                ran_num = random.randrange(0,100*valid_N)/(100*valid_N)
            validation_x.append(ran_num)

        validation_x.sort()
        for i in range(valid_N):
            for j in range(order+1):
                validation_X[i][j] = np.power(validation_x[i], j)
            t_validation.append(np.sin(2*pi*validation_x[i]))
        t_validation = np.array(t_validation)
        t_validation = t_validation.reshape(-1, 1)
        
        #测试数据集,评估模型效果
        test_x=[]
        t_test=[]
        test_X=np.ones((test_N, order+1))
        for i in range(test_N):
            ran_num = random.randrange(0,100*test_N)/(100*test_N)
            while(ran_num in x or ran_num in validation_x):
                ran_num = random.randrange(0,100*test_N)/(100*test_N)
            test_x.append(ran_num)
        
        test_x.sort()
        for i in range(test_N):
            for j in range(order+1):
                test_X[i][j] = np.power(test_x[i], j)
            t_test.append(np.sin(2*pi*test_x[i]))
        t_test = np.array(t_test)
        t_test = t_test.reshape(-1, 1)
        
        return (X, x, t, validation_X, validation_x, t_validation, test_X, test_x, t_test)

    def lossfunc(self, data, label, parameter, lamda):
        '''
        损失函数（带有正则项）
        data       :数据
        label      :标签
        parameter  :参数
        lamda      :正则系数
        '''
        mat = np.dot(data, parameter) - label
        result_mat = np.dot(mat.T, mat) + 0.5*np.dot(parameter.T, parameter)*lamda
        return result_mat[0][0]
    
    def conjugate_reg_gradient(self):
        '''
        共轭梯度法参数求解
        '''        
        #利用 (x^T*x)w=x^T*t 方程组的共轭梯度解法

        #令b=x^T*t
        #所求参数即为 Aw=b 中的解w
        b = np.dot(self.train_data.T, self.train_label)
        
        #正则项所用到的矩阵的对角线
        eye=[]
        #构造正则项所需的对角矩阵的对角线
        for i in range(self.order):
            if(i==0):
                eye.append(0)
            else:
                eye.append(1)
        eye.append(1)

        #正则项所需的对角矩阵
        reg_matrix=np.diag(eye)


        #正则项参数lamda,5种选择，从中挑选
        lamda_0 = 3e-7
        lamda = []
        for i in range(5):
            lamda.append(lamda_0*np.power(10, i))

        #存储在选取不同lamda情况下验证集上的最小二乘误差
        min_validation_loss = 1e7
        
        #遍历lamda的值，用验证集上的最小二乘误差确定最优的lamda取值
        for i in range(5):
            #对于每个lamda, 参数向量初始化为全0(列向量)
            w = [0]*(self.order+1)
            w = np.array(w)
            w = w.T
            w = w.reshape(self.order+1, 1)
            
            #加入正则项
            A = np.dot(self.train_data.T, self.train_data) + lamda[i]*reg_matrix
    
            #共轭梯度
            s = np.dot(A, w)
            #s = s.reshape(self.order+1, 1)
            #print('s:'+str(s.shape))
            r = b-s
            #print('r:'+str(r.shape))
            p = r
            #print('p:'+str(p.shape))
            for j in range(self.order+1):
                temp = np.dot(A, p)
                alpha = ((np.dot(r.T, p))/(np.dot(temp.T, p)))[0][0]
                '''
                print(alpha)
                print('w:' + str(w.shape))
                '''
                w = w + alpha*p
                '''
                print(w.shape)
                print(w)
                '''
                r = b - np.dot(A,w)
                if(not r.any()):
                    break
                beta = (-1)*(r.T@temp)/(p.T@temp)
                p = r+beta*p
    
            #求出验证集上的损失函数值
            temp_validation_loss=self.lossfunc(self.valid_data, self.valid_label, w, lamda[i])
    
            #挑选最小损失值的超参数lamda
            if(temp_validation_loss < min_validation_loss):
                lamda_flag = i
                min_validation_loss = temp_validation_loss
                w_final = w
        
            #把每一个超参lamda对应的图像都画出来
            w = np.array(w)
            w = np.squeeze(w)
            w = w[::-1]
            func_reg = np.poly1d(w)
            valid_reg = func_reg(self.valid_x)
            plt.title('Valid data with Regular Item: order=%d , datasize=%d, lamda=%.8f' %(self.order , len(self.valid_x), lamda[i]))
            plt.scatter(self.valid_x, self.valid_label)
            plt.plot(self.valid_x, valid_reg)
            plt.xlabel('x')
            plt.ylabel('validation_reg(x)')
            plt.show()

        #打印出最终选取的lamda取值
        print('选取的lamda为: %.8f' %lamda[lamda_flag])

        test_loss=self.lossfunc(self.test_data, self.test_label, w_final, lamda[lamda_flag])
        print('加正则项模型的测试集误差为: %f' %test_loss)

        #用测试集数据评估模型效果，画图
        w_t = w_final.T
        w_t = np.array(w_t)
        w_t = np.squeeze(w_t)
        w_t = w_t[::-1]
        func_reg = np.poly1d(w_t)
        test_reg = func_reg(self.test_x)
        plt.title('Test data with Regular Item: order=%d , datasize=%d, lamda=%.8f' %(self.order, len(self.test_x), lamda[lamda_flag]))
        plt.scatter(self.test_x, self.test_label)
        plt.plot(self.test_x, test_reg)
        plt.xlabel('x')
        plt.ylabel('y_reg(x)')
        plt.show()

if __name__ == '__main__':
    answer = conjugate_gradient()
    answer.conjugate_reg_gradient()

