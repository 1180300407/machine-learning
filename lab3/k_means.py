# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class k_means:
    '''
    k_means聚类算法
    '''
    def __init__(self, data, k):
        '''
        构造函数
        data:数据
        k   :聚类目标类别数
        '''
        self.data = data
        self.k = k
    
    #k—means算法初始化k个中心向量
    def initialize_center(self):
        #k个中心向量
        center = []
        #用集合来保存目前为止生成的随机中心id
        init_centerid = set();
        while(True):
            #当生成k个时，终止循环
            if(len(init_centerid)==self.k):
                break
            #生成一个[0,size-1]的索引，Data[rand_id]则为一个随机选定的中心
            rand_id = np.random.randint(0, self.data.shape[0]-1)
            #加入set中避免重复生成
            init_centerid.add(rand_id)
        
        for i in init_centerid:
            #读取集合中选取的索引id对应的向量，作为中心向量
            center_i = np.array(self.data[i])
            center.append(center_i)
        center = np.array(center)  
        return center
    
    #计算两个行向量间的距离
    #vector1:行向量1
    #vector2:行向量2
    def distance_calculate(self, vector1, vector2):
        minus = np.mat(vector1) - np.mat(vector2)
        distance = np.dot(minus, minus.T)
        return distance
    
    #重新选择k-means算法中每一类的中心向量
    #labels:目前算法为样本赋值的标签
    def re_center(self, labels):
        size = self.data.shape[0]
        dim = self.data.shape[1]
        #新的中心向量
        center_new = np.array([[float(0)]*dim]*self.k)
        center_new = center_new.reshape(self.k, dim)
        #记录每一个类别各有多少数据
        num = [0]* self.k
        for i in range(size):
            #第i个样本点所属类别为labels[i]
            #将其数据加到该类别中心向量上，最后中心向量求均值
            center_new[labels[i]] += self.data[i]
            num[labels[i]] += 1
        
        #之前中心向量保存的是其内样本数据之和，要求平均,作为新的中心向量
        for i in range(self.k):
            if(num[i]!=0):
                center_new[i] = center_new[i] / num[i]
                
        center_new=np.array(center_new)
        return center_new
    
    #k-means算法
    def k_means(self):
        size = self.data.shape[0]
        dim = self.data.shape[1]
        labels = [-1]*size
        center = np.array([[float(0)]*dim]*self.k)
        center = center.reshape(self.k, 2)
        center_new = self.initialize_center()
        #当中心向量不再变化时停止更新
        while(not((center==center_new).all())):
            center=center_new
        
            #每一个样本点
            for i in range(size):
                #维护每个样本点到所有中心向量的最小距离
                min_distance = 1e10
                #对所有中心向量都进行距离计算
                for j in range(self.k):
                    distance = self.distance_calculate(self.data[i], center[j])
                    #维护最小距离
                    if(distance<min_distance):
                        min_distance = distance
                        #对样本根据最小距离进行标签划分
                        labels[i] = j
                    
            center_new = self.re_center(labels)
    
        return labels,center

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


#可视化第index类的点，不同类别对应不同颜色
#Data :待可视化数据集
#size :数据集的大小
#index:数据集的分类类别
def draw(Data,size,index):
    x=[]
    y=[]
    #四种类别分用不同颜色画出
    color=['blue','green','yellow','orange']
    for i in range(size):
       x.append(Data[i][0]) 
       y.append(Data[i][1])
    plt.scatter(x,y,marker='o',c=color[index])   

#执行k_means算法进行聚类
def excute_kmeans():
    data, k = get_data()
    func = k_means(data, k)
    labels, center = func.k_means()
    #根据标签结果聚为k类，画出结果
    data_firsttype = []
    data_secondtype = []
    data_thirdtype = []
    data_fourthtype = []
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
    plt.title('labels-kmeans' )
    plt.show()
    
def main():
    excute_kmeans()
    
if __name__=='__main__':
    main()
    
