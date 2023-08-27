# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
def loadSimpData():
    """
    创建单层决策树的数据集
    Parameters:
        无
    Returns:
        dataMat - 数据矩阵
        classLabels - 数据标签
    """
    dataMat = np.matrix([[0., 1., 3.],
                      [0., 3., 1.],
                      [1., 2., 2.],
                      [1., 1., 3.],
                      [1., 2., 3.],
                      [0., 1., 2.],
                      [1., 1., 2.],
                      [1., 1., 1.],
                      [1., 3., 1.],
                      [0., 2., 1.]])
    classLabels = np.matrix([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    return dataMat, classLabels

def showDataSet(dataMat, labelMat):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    ax = plt.axes(projection='3d')
    data_plus = []  #正样本
    data_minus = [] #负样本
    labelMat = labelMat.T   #label矩阵转置
    #将数据集分别存放到正负样本的矩阵
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)      #转换为numpy矩阵
    data_minus_np = np.array(data_minus)    #转换为numpy矩阵
    ax.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], np.transpose(data_plus_np)[2], c='r')        #正样本散点图
    ax.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], np.transpose(data_minus_np)[2], c='b')     #负样本散点图
    plt.show()

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树分类函数
    Parameters:
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 划分的符号 - lt:less than，gt:greater than
    Returns:
        retArray - 分类结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 如果大于阈值,则赋值为-1
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    Tips:
    在已经写好单层决策树桩的前提下，这个函数需要用来确定哪个特征作为划分维度、划分的阈值以及划分的符号，从而输出“最佳的单层决策树”。
    具体来说，我们需要做一个嵌套三层的遍历：第一层遍历所有特征，第二层遍历这一维度特征所有可能的阈值，第三层遍历划分的符号；
    在确定以上三个关键信息之后，我们只需要调用决策树桩函数并获得其预测结果，结合真值计算误差；
    将误差最小的决策树桩的信息用一个字典储存下来，作为最终的输出结果；
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # 输出决策树信息，最小误差，估计的类别向量
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % \
                #         (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    完整决策树训练
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        numIt - 默认迭代次数
    Returns:
        weakClassArr- 完整决策树信息
        aggClassEst- 最终训练数据权值分布
    Tips:
    基于我们已经写好的最优决策树函数，我们可以在现有数据上进行迭代，不断更新数据权重、算法权重与其对应的决策树桩，
    直到误差为零，退出循环
    """
    weakClassArr = []  # 用于存储弱分类器
    m = np.shape(dataArr)[0]  # 数据点数量
    D = np.mat(np.ones((m, 1)) / m)  # 初始化权值矩阵
    aggClassEst = np.mat(np.zeros((m, 1)))  # 记录每个数据点的类别估计累计值

    for i in range(numIt):
        # 1. 使用buildStump()函数得到当前迭代中最佳的弱分类器
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)

        # 2. 计算当前弱分类器的权重alpha，防止过拟合，需要控制alpha小于1.0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        # 3. 更新权值向量D
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # 4. 计算每个数据点的类别估计累计值
        aggClassEst += alpha * classEst

        # 5. 计算错误率，如果错误率为0，则直接跳出循环
        errorRate = np.sum(np.sign(aggClassEst) != np.mat(classLabels).T) / m
        if errorRate == 0.0:
            break

    # 返回弱分类器集合和每个数据点的类别估计累计值
    return weakClassArr, aggClassEst
if __name__ == '__main__':
    dataArr, classLabels = loadSimpData()
    showDataSet(dataArr, classLabels)
    weakClassArr,aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    print(weakClassArr)
    print(aggClassEst)
