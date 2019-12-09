#coding:utf-8
#Author:Dustin
#Algorithm:决策树(ID3)

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：82.3%
    运行时长：371s
'''

from keras.datasets import mnist
import numpy as np
import time
from itertools import chain
import copy

class DecisionTree:
    #定义初始化函数，设置决策树信息增益阈值。
    def __init__(self, threshold):
        self.epsilon = threshold

    #计算经验熵
    def __getHD(self, label):
        HD = 0
        label_set = set([i for i in label])
        for em in label_set:
            p = label[label == em].size / label.size
            HD += - 1 * p * np.log(p)
        return HD

    #计算经验条件熵
    def __getHDA(self, data, label):
        HDA = 0
        feature_set = set([i for i in data])
        for i in feature_set:
            HDA += (data[data == i].size / data.size) * self.__getHD(label[data == i])
        return HDA
    
    #计算label中出现次数最多的标签
    def __maxClass(self, label):
        freq_count = {key: label.count(key) for key in set(label)}
        freq_count_e = {v:k for (k, v) in freq_count.items()}
        return freq_count_e.get(max(freq_count_e))
        
    #计算信息增益，获取其最大值对应的特征选择。
    def __bestFeature(self, data, label):
        data = np.array(data)
        label = np.array(label).T
        feature_num = np.shape(data)[1]
        max_gain = -1
        max_feature = -1
        for feature in range(feature_num):
            HD = self.__getHD(label)
            data_feature = np.array(data[:, feature].flat)
            HDA = self.__getHDA(data_feature, label)
            gain = HD - HDA
            if gain > max_gain:
                max_gain = gain
                max_feature = feature
        return max_feature, max_gain
            
    #去掉特征Ag，划分子集。
    def __subDataSet(self, data, label, Ag, value):
        subData = []
        subLabel = []
        for i in range(len(data)):
            if (data[i][Ag] == value):
                subData.append(np.append(data[i][0:Ag], data[i][Ag + 1 :]))
                subLabel.append(label[i])
        return subData, subLabel

    #建立决策树
    def __createTree(self, *dataset):
        train_data = dataset[0][0]
        train_label = dataset[0][1]

        m = np.shape(train_data)
        n = np.shape(train_label)

        print('start a node', m, n)
        epsilon = self.epsilon
        label_dict = {em for em in train_label}

        if len(label_dict) == 0:
            return 0

        if len(label_dict) == 1:
            return train_label[0]

        if len(m) == 1:
            return self.__maxClass(train_label)
        
        Ag, epsilon_get = self.__bestFeature(train_data, train_label)
        if epsilon_get < epsilon:
            return self.__maxClass(train_label)

        tree = {Ag: {}}
        for i in range(256):
            tree[Ag][i] = self.__createTree(self.__subDataSet(train_data, train_label, Ag, i))

        return tree

    #定义fit函数，根据训练集创建决策树。
    def fit(self, *trainset):
        self.tree = self.__createTree(trainset)

    #获取样本对应的预测标签
    def __getLabel(self, data):
        tree = copy.deepcopy(self.tree)
        while True:
            if (type(tree) == np.uint8):
                return tree
            (key, value), = tree.items()
            if type(tree[key]).__name__ == 'dict':
                key_data = data[key]
                np.delete(data, key)
                tree = value[key_data]
                if type(tree).__name__ == 'int':
                    return tree
            else:
                return value
    #定义预测函数
    def predict(self, test_data):
        tree = self.tree
        predict_label = []
        for i in range(len(test_data)):
            predict_label.append(self.__getLabel(test_data[i]))
        return predict_label

    #定义准确率计算函数
    def score(self, test_data, test_label):
        predict_label = self.predict(test_data)
        error = 0
        length = len(predict_label)
        for i in range(length):
            if (predict_label[i] != test_label[i]):
                error += 1
        return 1 - error / length


if __name__ == '__main__':
    (train_data, train_label), (test_data, test_label) = mnist.load_data()  #加载mnist数据
    train_data = [np.array(data).flatten().tolist() for data in train_data[:1000]]  #将28*28维矩阵转换为1*784的矩阵。
    train_label = train_label[:1000]
    test_data = [np.array(data).flatten().tolist() for data in test_data[:100]]
    test_label = test_label[:100]
    dt = DecisionTree(threshold=0.1) #初始化决策树分类器
    start = time.time()
    dt.fit(train_data, train_label) #构造决策树
    print("决策树预测准确率：%.2f%%" % (dt.score(test_data, test_label) * 100))
    end = time.time()
    print("耗时：%.2f s" %(end - start))