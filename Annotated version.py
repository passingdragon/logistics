import lr_utils
import numpy as np
import numba as nb

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
# 化成二维数据
# 标准化
# sigmoid
# initial
# propagate 计算dw,db
# optimize 迭代求w,b
# predict 预测数据集 得到Y_P
# logistic 封装
train_set_x_f = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_f = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# 将矩阵由四维的变成2维的，一个图片用一列数字来表示（n,n,3,1）变为（n*n*3,1)

train_set_x = train_set_x_f / 255
test_set_x = test_set_x_f / 255


# 广播，对数据进行标准化，一般都是除以范数，这里偷懒，已经知道每个数都小于等于255
def sigmoid(X):
    s = 1 / (1 + np.exp(-X))
    return s


# 定义sigmoid函数

def initial(dim):
    # dim 为一个测试用例的维数 ， 训练集为（dim,1)
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    # 确保w的维数不会出错
    return w, b


# 初始化参数

def propagate(w, b, X, Y):
    A = sigmoid(np.dot(w.T, X) + b)
    m = X.shape[1]
    # cost = (-1 / m) * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    # 向后传播计算dw,db
    assert (dw.shape == w.shape)
    # 确定一下dw维数
    gards = {"dw": dw,
             "db": db}
    return gards


# 计算一次传播dw,db
def optimize(w, b, X, Y, num=2000, beta=0.5):
    # num为训练次数
    # beta为学习率
    # X为训练集
    # Y为训练集的正确答案
    for i in range(num):
        g = propagate(w, b, X, Y)
        dw = g["dw"]
        db = g["db"]
        w = w - beta * dw
        b = b - beta * db
    param = {"w": w,
             "b": b}
    return param


# 训练得到合适的参数w,b

def predict(w, b, X):
    w = w.reshape((X.shape[0], 1))
    # 确定维度
    y_P = sigmoid(np.dot(w.T, X) + b)
    Y_P = np.zeros((1, X.shape[1]))
    for i in range(y_P.shape[1]):
        if y_P[0][i] <= 0.5:
            Y_P[0, i] = 0
        else:
            Y_P[0, i] = 1
    assert (Y_P.shape == (1, X.shape[1]))
    return Y_P


# 进行0,1预测

def logistic(x_train, y_train, x_test, y_test, num=2000, beta=0.5):
    w, b = initial(x_train.shape[0])
    # 初始化
    params = optimize(w, b, x_train, y_train, num, beta)
    # 迭代得到参数
    w = params["w"]
    b = params["b"]
    Y_train_P = predict(w, b, x_train)
    Y_test_P = predict(w, b, x_test)
    # 预测
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_P - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_P - y_test)) * 100))
    # 得到正确率
    # np.mean 是求均值，可自行查阅资料了解
    ans = {"w": w,
           "b": b,
           "y_test_P": Y_test_P,
           "y_train_P": Y_train_P}
    return ans

ans = logistic(train_set_x, train_set_y, test_set_x, test_set_y)
