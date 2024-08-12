---
title: 用python实现感知机来给鸢尾花分类
date: '2024-04-22 18:26:40'
updated: '2024-08-12 21:36:27'
permalink: >-
  /post/use-python-to-implement-the-perceptual-machine-to-classify-the-iris-2wnftj.html
comments: true
toc: true
---

# 用python实现感知机来给鸢尾花分类

# 了解鸢尾花数据集

鸢尾花数据集是机器学习中常用的一个数据集，它包含了山鸢尾、变色鸢尾和弗吉尼亚鸢尾三种鸢尾属植物的150朵花的测量结果，数据中的每列分别为：

* 萼片长度
* 萼片宽度
* 花瓣长度
* 花瓣宽度
* 分类标签

# 感知机

感知机仅由一个神经元组成，是一种非常简单的分类算法，它可以用来分类可以被一个超平面分割的数据。

# 代码实现

## 引入所需的库

```python
import pandas as pd  
import numpy as np
```

## 实现感知机对象

```python
class Perceptron:  
  
    def __init__(self, eta=0.01, iter_num=10, random_state=1):  
        self.w = None  
        self.eta = eta  
        self.iter_num = iter_num  
        self.random_state = random_state  
  
    def train(self, xx, yy):  
        self.w = np.random.RandomState(self.random_state).normal(loc=0.0,  
                                                                 scale=0.01,  
                                                                 size=1 + xx.shape[1])  
        train_errors = []  
        for i in range(self.iter_num):  
            error = 0  
            for data, target in zip(xx, yy):  
                data = np.insert(data, 0, 1)  
                update = self.eta * (target - self.input(data))  
                self.w += update * data  
                if update != 0:  
                    error += 1  
            train_errors.append(error)  
        return train_errors  
  
    def input(self, data):  
        return 1 if np.dot(data, self.w) >= 0.0 else -1  
  
    def predict(self, data):  
        data = np.insert(data, 0, 1)  
        return self.input(data)
```

在构造函数中，我们传入`eta`​，`iter_num`​，`random_state`​三个参数。

* ​`eta`​：学习率
* ​`iter_num`​：训练次数
* ​`random_state`​：随机数种子，这里传入一个固定的值来使每次运行的结果都一样

我们实现了`train`​、`input`​和`predict`​三个方法。

​`train`​方法中，先调用numpy按照标准差为0.01的正态分布生成随机数，作为权重w的初始值。  
接下来是训练过程，这里给数据前补了一个1，是因为我们把偏置作为w的第一维，补1后就可以直接用向量相乘的方法计算模型的结果。  
train_errors中存储了每次训练错误的数量，用于直观显示模型结果。

​`input`​方法根据模型计算的结果绝对传入的是正类还是负类。

​`predict`​方法用于外部调用。

## 利用鸢尾花数据集训练

```python
if __name__ == '__main__':  
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'  
    df = pd.read_csv(s, header=None, encoding='utf-8')  
    x = df.iloc[:100, [0, 1, 2, 3]].values  
	y = df.iloc[:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)  
    perceptron = Perceptron()  
    errors = perceptron.train(x, y)  
    print(errors)
```

这里直接从UCI的服务器上获取csv文件，用pandas的dataframe存储。再用iloc方法取出前100行中的指定列，然后用values方法得到numpy数组。

这里读取前100行的原因是，感知机是一种二分类模型，无法对三种标签的数据分类。  
我们可以用OvA（one-versus-all，一对多）技术分类，即为每个类都训练一个感知器，属于这个类的是正类，其余的是负类，这里不再赘述。

因为原数据中分类标签分别是Iris-setosa、Iris-versicolor，所以这里用numpy的where方法把标签改为-1和1。

## 运行结果

运行结果如下：

​![用python实现感知机来给鸢尾花分类](https://raw.githubusercontent.com/z131f/z131f.github.io/main/images/%E7%94%A8python%E5%AE%9E%E7%8E%B0%E6%84%9F%E7%9F%A5%E6%9C%BA%E6%9D%A5%E7%BB%99%E9%B8%A2%E5%B0%BE%E8%8A%B1%E5%88%86%E7%B1%BB-20240422182617-2ik22bg.png)​

可以看到，从第五次之后，模型的预测不再有错误，即训练完毕。
