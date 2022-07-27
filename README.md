### 本项目是利用Weka框架API实现数据分析和数据挖掘的相关案例。

+++

### 实验环境

- JDK版本11.0.15
- Weka版本为3.8.6
- Win11系统，IDEA开发环境

### 实验内容

实验主体分为回归、聚类、分类以及关联规则挖掘三部分。

1、分类

利用决策树J48模型、随机森林模型、神经网络模型、朴素贝叶斯算法等对数据进行分类，其中60%作为训练集，40%作为测试集。

由结果计算查准率、查全率、混淆矩阵和运行时间

2、回归

利用随机森林、神经网络、线性回归的方法对数据进行回归分析，其中60%作为训练集，40%作为测试集。

由结果计算均方根误差和相对误差

3、聚类

分别使用K均值、EM、层次聚类对数据进行聚类。通过轮廓系数和SDbw指标对聚类效果进行评估

4、关联规则分析

在置信度为0.3、0.6和0.9的情况下对数据进行关联规则的分析，得到相应的规则集。