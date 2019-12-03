#coding:utf-8
#Author:Dustin
#Algorithm:K近邻

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000 (实际使用100，否则运行时间会很长Orz)
------------------------------
运行结果：（k_neighbor：25）
向量距离度量值——欧式距离
    正确率：83%
    运行时长：344.58s
向量距离度量值——曼哈顿距离
    正确率：56%
    运行时长：134.85s
'''

from keras.datasets import mnist
import numpy as np
import time
from itertools import chain

class KNN:
    #定义初始化方法，记录分类器K近邻参数k_neighbor和Lp范数中的参数p。
    def __init__(self, k_neighbor, p):
        self.k = k_neighbor
        self.p = p

    #计算两点之间的距离，当p为1时计算曼哈顿距离，当p为2时计算欧式距离。
    def distance(self, x1, x2):
        p = self.p
        return np.linalg.norm(x1-x2, ord=p)

    #针对x，检测与训练集中距离最近的k个样本，将k个样本中出现次数最多的标签作为x的预测标签值。
    def getLabel(self, train_data, train_label, x):
        train_data = np.mat(train_data) #将列表转为矩阵，方便运算。
        train_label = np.mat(train_label).T
        k = self.k
        m, n = np.shape(train_data)
        dist = np.zeros(m) 
        for i in range(m):
            y = train_data[i]
            dist[i] = self.distance(x, y)
        kNearest = np.argsort(np.array(dist))[:k] #kNearest列表中存储了k个样本在train_data中的索引值
        vote = [0] * 10 #将投票计数器置零
        for i in kNearest: #针对每一个k近邻，获取样本的标签train_label[i]，将其作为vote列表下标，并将vote[train_label[i]]加1
            vote[int(train_label[i])] += 1
        return vote.index(max(vote)) #返回投票计数器最大值对应的索引，即出现次数最多的标签。

    #预测测试集样本的标签值
    def predict(self,train_data, train_label, test_data):
        print("开始预测")
        num = len(test_data)
        predict_label = []
        for i in range(num):
            predict_label.append(self.getLabel(train_data, train_label, test_data[i]))
            print('\r迭代进度|%-50s| [%d/%d]' % ('█' * int((i / num) * 50 + 2), i + 1, num), end='') #绘制进度条
        print("\n结束预测")
        return np.array(predict_label)

    #将predict函数返回的测试集样本预测标签与测试集验证标签进行对比，计算KNN算法准确率。
    def score(self, train_data, train_label, test_data, test_label):
        predict_label = self.predict(train_data, train_label, test_data)
        num = len(test_label) #计算测试集样本标签总数
        error = 0 #错误值计数器
        for i in range(num):
            if (predict_label[i] != test_label[i]):
                error += 1
        accuracy = 1 - (error / num)
        return accuracy

if __name__ == '__main__':
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = np.array([list(chain(*i)) for i in train_data]) #将28*28维矩阵转换为1*784的向量
    test_data = np.array([list(chain(*i)) for i in test_data[:100]]) #对测试集进行切片，选取前100组数据。
    test_label = test_label[:100]
    knn = KNN(k_neighbor=25, p=2) #设置KNN参数，p=2表示使用欧式距离进行度量。
    start = time.time()
    print("K近邻预测准确率：%.2f%%" % (knn.score(train_data, train_label, test_data, test_label) * 100))
    end = time.time()
    print("耗时：%.2f s" %(end - start))
    



