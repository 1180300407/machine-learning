# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

class PCA_maxVariance:
    def __init__(self, data, data_size, dim, target_dim, data_flag=True):
        '''
        data      :样本数据
        data_size :样本数据集的大小
        dim       :样本数据的维度
        target_dim:将样本数据进行PCA降维的目标维度
        data_flag :标识数据是否采用自动生成的3D数据，默认为true，即采用
        '''
        self.data_size = data_size
        if data_flag:
            self.data = self.get_3Ddata()
        else:
            self.data = data
        self.dim = dim
        self.target_dim = target_dim


    #生成三维数据，且数据主要分布在低维空间中(2维)
    #Datasize:生成数据集的大小
    def get_3Ddata(self):
        #三个维度的均值
        mean = [1, 2, 2]
        #三个维度的方差，其中第一维度远小于其它维度，使数据主要分布在低维空间
        cov = [[10, 0, 0], [0, 10, 0], [0, 0, 0.0005]]
        #保存生成的数据
        data = []
        for index in range(self.data_size):
            data.append(np.random.multivariate_normal(mean, cov).tolist())
    
        return np.array(data)

    #计算样本数据的均值与向量与协方差矩阵
    def cal_meanAndcov(self):
        #样本的均值向量求解
        mean = []
        #计算第i个维度的均值
        for i in range(self.dim):
            sum_i = 0
            for j in range(self.data_size):
                sum_i += self.data[j][i]
            mean_i = float(sum_i) / self.data_size
            mean.append(mean_i)
        mean = np.array(mean)
        #1*dim的行向量
        mean = mean.reshape((1, self.dim))
    
        #将均值扩展为Datasize*dim的矩阵，来实现Data-mean矩阵减法
        mean_mat = np.tile(mean, (self.data_size, 1))
        Data_minus = self.data-mean_mat
    
        #协方差矩阵
        cov = 1.0 / self.data_size * np.dot(Data_minus.T, Data_minus)
    
        #返回均值与协方差
        return mean, cov

    #寻找value中的最大值对应的下标index
    #value :数据集
    #length:数据集的大小
    def getindex_maxvalue(self, value, length):
        max_id = 0
        max_value = value[0]
        for i in range(length):
            if(value[i] > max_value):
                max_id = i
                max_value = value[i]
    
        return max_id

    #对数据进行PCA降维
    def PCA(self):
        #得到样本的均值与协方差矩阵
        mean, cov = self.cal_meanAndcov()
    
        #求解协方差矩阵的特征值与特征向量
        character_value, character_vector = np.linalg.eig(cov)
        character_value = character_value.reshape((self.dim, 1))
        print("%d 个原始特征向量:" %self.dim)
        print(character_vector)
        print("对应的特征值为:")
        print(character_value)
    
        #保存降维后最终保留的特征向量
        retain_vector=[]
    
        #保存未保留的特征向量，用来最小化误差
        delete_vector=[]
    
        #根据特征值的大小，选取保留的特征向量(target_dim个较大的特征值对应的特征向量)
        for i in range(self.target_dim):
            index = self.getindex_maxvalue(character_value, len(character_value))
            retain_vector.append(character_vector[index])
            #将选取的特征值去除
            character_value = np.delete(character_value, index, axis=0)
            character_vector = np.delete(character_vector, index, axis=0)
        
        #未保留的特征向量
        delete_vector = character_vector
    
        #target_dim个保留的特征向量
        retain_vector = np.array(retain_vector)
    
        #dim-target_dim个未保留特征向量
        delete_vector = np.array(delete_vector)
    
        #求取每个数据点在每个特征向量下的投影(坐标值)
        Data_projection = np.dot(self.data, retain_vector.T)
    
        #为了最小化误差而加上的偏置
        bias = np.dot(mean, delete_vector.T)
    
        return Data_projection, bias, retain_vector, delete_vector
    
    #对投影后的二维数据进行画图展示
    #data_projection :待展示的2维数据
    def draw2D_PCAdata(self, data_projection):
        x=[]
        y=[]
        for i in range(self.data_size):
            x.append(data_projection[i][0])
            y.append(data_projection[i][1])
        plt.scatter(x, y, marker='o', c="blue")
        plt.show()
    
    #对于3D数据进行画图展示，注意如果数据维度不为3，it will do nothing
    def draw3D_data(self, image_label):
        if self.dim != 3:
            return
        fig = plt.figure()
        ax = Axes3D(fig)
        x=self.data[:,0]
        y=self.data[:,1]
        z=self.data[:,2]
        ax.scatter(x,y,z,facecolor="r", edgecolor="b", label=image_label)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    
        ax.plot(x,y,'y+',zdir='z')
        plt.legend()
        plt.show()
        return 

    #对pca降维后的数据进行重建
    #pca_data     :降维后数据
    #retain_vector:pca保留的主成分
    #delete_vector:pca未保留的成分向量
    #bias         :pca未保留成分向量的系数
    def rebulid(self, pca_data, retain_vector, delete_vector, bias):
        #数据集大小
        Datasize = pca_data.shape[0]
        #保存重建的数据
        Data_rebuild = np.dot(pca_data, retain_vector)
        #利用保存的bias偏差减小还原时的误差
        vector_bias = np.dot(bias, delete_vector)
        mat_bias = np.tile(vector_bias, (Datasize, 1))
        #重建
        Data_rebuild = Data_rebuild + mat_bias
        return Data_rebuild
        
def main():
    # 用于生成数据的测试
    dim = 3
    N = 50
    pca = PCA_maxVariance('', N, dim, dim-1)
    pca.draw3D_data('Origin Data')
    data_projection, bias_pca, vector_retain, vector_delete = pca.PCA()
    print("Retain vectors:")
    print(vector_retain)
    print("Data_projection:")
    print(data_projection)
    print("bias:")
    print(bias_pca)

    pca.draw2D_PCAdata(data_projection)

    '''
    print('Origin Data:')
    print(pca.data)
    '''
    
    '''
    data_rebuild = pca.rebulid(data_projection, vector_retain, vector_delete, bias_pca)
    print('Rebuild Data:')
    print(data_rebuild)
    '''
    pca.draw3D_data('Rebuild Data')

if __name__ == '__main__':
    main()
