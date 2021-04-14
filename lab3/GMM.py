# -*- coding: utf-8 -*-
from k_means import k_means,draw
import numpy as np
import matplotlib.pyplot as plt

class GMM:
    '''
    GMM聚类
    '''
    def __init__(self, center, labels, k_means_flag, terminal, data, k):
        '''
        构造函数
        center      :预先设置的中心向量
        labels      :预先设置的标签
        k_means_flag:是否使用k_means作为初始化的标识
        terminal    :迭代次数
        data:       :数据
        k           :聚类目标类别数
        '''
        self.data = data
        self.k = k
        self.terminal = terminal
        if k_means_flag:
            func = k_means(data, k)
            self.labels, self.center = func.k_means()
        else:
            self.labels = labels
            self.center = center

    #GMM初始化均值向量与协方差,以及隐变量的离散概率分布alpha
    def GMM_init(self):
        #初始化均值向量为k-means的分类中心向量
        mean = self.center
        #协方差
        cov = []
        
        size = self.data.shape[0]
        #计算k个类别(k个高斯分布)的初始协方差
        for i in range(self.k):
            #存储类别为第i类的数据样本
            data_i = []
            for j in range(size):
                if(self.labels[j] == i):
                    data_i.append(self.data[j,:])
            length = len(data_i)
            data_i = np.array(data_i)
        
            #用矩阵点乘计算协方差，需要计算(X1-E(X1))以及(X2-E(X2))，为length*2维向量
            #因此需要把第i类的均值扩展成length*2
            temp_mean = []
            for l in range(length):
                temp_mean.append(mean[i,:])
            temp_mean = np.array(temp_mean)
            #第i类的协方差
            cov_i = np.dot((data_i-temp_mean).T,(data_i-temp_mean))/length
            cov.append(cov_i)
        cov = np.array(cov)
        alpha = []
        for i in range(self.k):
            alpha.append(float(1)/self.k)
        return mean, cov, alpha


    #在index_label类的均值mean，方差cov的二维高斯分布条件下，计算第index_data个数据样本的概率
    #data      :数据
    #mean      :均值向量
    #cov       :协方差
    def Gauss_PDF(self, data, mean, cov):
        data = np.array(data)
        dim = len(data)
        data = data.reshape(dim, 1)
        mean = np.array(mean)
        mean = mean.reshape(dim, 1)
        power = np.exp(-1 / 2 * np.dot((data - mean).reshape(1, dim), np.linalg.inv(cov).dot((data - mean).reshape(dim, 1))))
        temp = pow(2 * np.pi, dim / 2) * pow(np.linalg.det(cov), 0.5)
        return power[0][0] / temp



    #用EM迭代算法实现GMM模型(采用k-means结果进行初始化)
    def GMM_EM(self):
        #利用k—means结果得到初始参数
        init_value = self.GMM_init()
        mean = init_value[0]
        cov = init_value[1]
        alpha = init_value[2]
        
        n, m = self.data.shape
    
        gamma = np.zeros((n, self.k))
        
        # EM 算法
        for step in range(self.terminal):
            # E-step
            for i in range(n):
                #temp列表中的每一项对应于原函数的一个sum_latent
                
                temp = []
                for j in range(self.k):
                    #print('data_i: ' + str(self.data[i]))
                    #print('mean: ' + str(mean[j]))
                    #print('cov: ' + str(cov[j]))
                    temp.append(alpha[j] * self.Gauss_PDF(self.data[i], mean[j], cov[j]))
               
                #求和
                sum_temp = sum(temp)
                #更新gamma矩阵
                for j in range(self.k):
                    gamma[i][j] = temp[j] / sum_temp

            # M-step
            temp = [sum([gamma[i][j] for i in range(n)]) for j in range(self.k)]
            for j in range(self.k):
                #更新均值
                mean[j] = sum([gamma[i][j] * self.data[i] for i in range(n)]) / temp[j]
                #更新协方差
                cov[j] = sum([gamma[i][j] * np.dot((self.data[i] - mean[j]).reshape(m, 1), (self.data[i] - mean[j]).reshape(1, m))
                      for i in range(n)]) / temp[j]
                #更新先验概率
                alpha[j] = temp[j] / n
        return gamma, mean
        
        

    #根据GMM生成的隐变量概率矩阵进行分类 
    #gamma:GMM生成的隐变量概率矩阵
    def GMM_EM_labels(self, gamma):
        size = self.data.shape[0]
        
        #分类标签
        labels=[0]*size
    
        #寻找每个样本在不同类别下的最大概率
        for i in range(size):
            probability=0
            kind=0
            for j in range(self.k):
                if(gamma[i][j]>probability):
                    probability=gamma[i][j]
                    kind=j
            labels[i]=kind
        return labels

#生成数据，这里采取N=200,k=4
def get_data():
    #样本个数
    #N=200
    N1=50
    N2=50
    N3=50
    N4=50
    #聚类的类别个数
    k=4

    #k个高斯分布的均值方差
    mean1=[1,1]
    sigma=np.mat([[1,0],[0,1]])
    mean2=[4,1]
    mean3=[1,4]
    mean4=[4,4]

    #生成N个样本数据
    data1=np.random.multivariate_normal(mean1,sigma,N1)
    data2=np.random.multivariate_normal(mean2,sigma,N2)
    data3=np.random.multivariate_normal(mean3,sigma,N3)
    data4=np.random.multivariate_normal(mean4,sigma,N4)
    data=np.vstack((data1,data2,data3,data4))

    #将点集以及真实类别情况画出来
    draw(data1,N1,1-1)
    draw(data2,N2,2-1)
    draw(data3,N3,3-1)
    draw(data4,N4,4-1)
    plt.title('true labels' )
    plt.show()
    return data, k
    
def main():
    #设置GMM_EM迭代次数
    terminal = 1000
    data, k = get_data()
    gmm = GMM('', '', True, terminal, data, k)
    
    #执行GMM_EM
    GMM_value = gmm.GMM_EM()
    #用EM算法得到的GMM模型的概率矩阵以及均值向量
    gamma = GMM_value[0]
    #mean = GMM_value[1]
    #用得到的概率矩阵以及均值向量进行软分类
    labels = gmm.GMM_EM_labels(gamma)

    #根据标签结果聚为k类，画出结果
    data_firsttype=[]
    data_secondtype=[]
    data_thirdtype=[]
    data_fourthtype=[]

    N = data.shape[0]
    for i in range(N):
        if(labels[i]==0):
            data_firsttype.append(data[i])
        elif(labels[i]==1):
            data_secondtype.append(data[i])
        elif(labels[i]==2):
            data_thirdtype.append(data[i])
        else:
            data_fourthtype.append(data[i])
    draw(data_firsttype,len(data_firsttype),0)
    draw(data_secondtype,len(data_secondtype),1)
    draw(data_thirdtype,len(data_thirdtype),2)
    draw(data_fourthtype,len(data_fourthtype),3)
    plt.title('labels-GMM_EM' )
    plt.show()

if __name__=='__main__':
    main()