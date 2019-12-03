#coding:utf-8
#Author:Dustin
#Algorithm:单层感知机(二分类)

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
正确率：80.29%（二分类）
运行时长：78.55s
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
    def fit(self, train_data, train_label):
        print("开始训练")
        data = np.mat(train_data)  #转换为矩阵，后面的运算会更方便。实际上，在转换为矩阵后运算符重载了。
        label = np.mat(train_label).T  #将标签矩阵转置
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
    
    #定义predict方法，读取测试集，返回预测标签。
    def predict(self, test_data):
        print("开始预测")
        data = np.mat(test_data)
        m, n = np.shape(data)
        predict_label = []  #定义存储预测标签的列表
        w = self.w  #读取fit后的w和b
        b = self.b

        for i in range(m):  #对每一个样本进行检测
            xi = data[i]
            result = np.sign(w * xi.T + b)
            predict_label.append(result)
        
        print("结束预测")
        
        predict_label = np.array(predict_label)
        return predict_label  #返回预测标签值

    #定义score函数，返回预测准确率。
    def score(self, test_data, test_label):
        predict_label = np.mat(self.predict(test_data)).T
        test_label = np.mat(test_label).T
        m, n = np.shape(test_label)
        error = 0
        for i in range(m):
            if (predict_label[i] != test_label[i]):
                error += 1
        accuracy = 1 - (error / m)
        return accuracy


if __name__ == '__main__':
    #对数据进行预处理，将每一个样本的图片数据由28*28的矩阵转换为1*784的矩阵。
    #由于单层感知机只能处理二分类的情况，所以需要对标签进行二值化。
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = np.array([list(chain(*i)) for i in train_data])
    train_label = np.array([1 if i >= 5 else - 1 for i in train_label])
    test_data = np.array([list(chain(*i)) for i in test_data])
    test_label = np.array([1 if i >= 5 else - 1 for i in test_label])
    
    #对训练和测试过程进行计时
    start = time.time()
    pc = Perceptron(iteration=30, learning_rate=0.001)
    pc.fit(train_data, train_label)
    print("单层感知机预测准确率：%.2f%%" % (pc.score(test_data, test_label)*100))
    end = time.time()
    print("耗时：%.2f s" %(end - start))

