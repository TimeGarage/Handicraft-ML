#coding:utf-8
#Author:Dustin
#Algorithm:单层感知机

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
正确率：80.29%（二分类）
运行时长：67.88s
'''

from keras.datasets import mnist
import numpy as np
from itertools import chain
import time

class Perceptron:
    #定义初始化方法，记录迭代次数和学习率。
    def __init__(self, iteration = 30, learning_rate = 0.001):
        self.iteration = iteration
        self.rate = learning_rate
    #定义fit方法，使用训练集完成参数w和b的训练。
    def fit(self, data, label):
        print("开始训练")
        data = np.mat(data)  #转换为矩阵，后面的运算会更方便。实际上，在转换为矩阵后运算符重载了。
        label = np.mat(label).T  #将标签矩阵转置
        m, n = np.shape(data) #获取数据行列数
        w = np.zeros((1, n)) #初始化w矩阵
        b = 0  #初始化偏置项b
        iteration = self.iteration
        rate = self.rate

        for i in range(iteration): #迭代iteration次
            for j in range(m): #每次迭代使用m组数据更新参数，m在fit方法中即训练集样本数。
                xi = data[j] #选取单个样本所对应的矩阵
                yi = label[j] #选取样本标签
                result = -1 * yi * (w * xi.T + b) #使用梯度下降法求解参数w和b
                if result >= 0:
                    w += rate * (yi * xi) #注意yi和xi的顺序，只有yi在前才能保证结果维度的正确性。
                    b += + rate * yi
            print('\r迭代进度|%-50s| [%d/%d]' % ('█' * int((i / iteration) * 50 + 2),
                  i + 1, iteration), end='') #绘制进度条
        
        self.w = w  #更新参数w和b
        self.b = b
        print("\n结束训练")
    
    #定义predict方法，读取测试集，返回预测标签，同时更新预测准确率。
    def predict(self, data, label):
        print("开始测试")
        data = np.mat(data)
        label = np.mat(label).T
        m, n = np.shape(data)
        error = 0  #定义error对预测错误的点进行计数
        label_res = []  #定义存储预测标签的列表
        w = self.w  #读取fit后的w和b
        b = self.b

        for i in range(m):  #对每一个样本进行检测
            xi = data[i]
            yi = label[i]
            result = -1 * yi * (w * xi.T + b)
            label_res.append(result)
            if result >= 0:
                error += 1
        
        self.__accuracy = 1 - error / m  #计算准确率
        print("结束测试")
        return label_res  #返回预测标签值

    #定义score函数返回模型本次测试的准确率
    def score(self):
        return self.__accuracy 


if __name__ == '__main__':
    #对数据进行预处理，将每一个样本的图片数据由28*28的矩阵转换为1*784的矩阵。
    #由于单层感知机只能处理二分类的情况，所以需要对标签进行二值化。
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = np.array([list(chain(*i)) for i in train_data])
    test_data = np.array([list(chain(*i)) for i in test_data])
    train_label = np.array([1 if i >= 5 else - 1 for i in train_label])
    test_label = np.array([1 if i >= 5 else - 1 for i in test_label])
    
    #对训练和测试过程进行计时
    start = time.time()
    pc = Perceptron(iteration=30, learning_rate=0.001)
    pc.fit(train_data, train_label)
    pc.predict(test_data, test_label)
    end = time.time()
    print("单层感知机预测准确率：%2f%%" % (round(pc.score(), 4) * 100))
    print("耗时：%3f s" %(end - start))

