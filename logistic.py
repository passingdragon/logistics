import lr_utils
import numpy as np
import numba as nb
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# 处理数据
# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
# train_set_x_flatten维度：（12288，209）
# trainsety 维度：（1，209）
# test_set_x_flatten维度：（12288，50）
# test_set_y维度：（1，50）

# 数据标准化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

@nb.jit()
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

@nb.jit()
def initial(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b

@nb.jit()
def propagate(w, b, X, Y):
    A = sigmoid(np.dot(w.T, X) + b)
    m = X.shape[1]
    cost = (-1 / m) * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)

    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost

@nb.jit()
def optimize(w, b, X, Y, num = 2000, beta = 0.5):
    # num is number of loop
    # beta is 学习率
    costs = []
    for i in range(num):
        g, cost = propagate(w, b, X, Y)
        dw = g["dw"]
        db = g["db"]
        w = w - beta * dw
        b = b - beta * b
        # if i % 100 == 0:
        #     costs.append(cost)
    param = {"w": w,
             "b": b}
    grads = {"dw": dw,
             "db": db}
    return param, grads, costs

@nb.jit()
def predict(w, b, X):
    m = X.shape[1]
    Y_P = np.zeros((1, m))
    w = w.reshape((X.shape[0], 1))
    # 确保维度相同，能够相乘

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0][i] <= 0.5:
            Y_P[0, i] = 0
        else:
            Y_P[0, i] = 1
    assert (Y_P.shape == (1, m))
    return Y_P
@nb.jit()
def logistic(X_train,Y_train,X_test,Y_test,num = 2000,beta = 0.5):
    w,b = initial(X_train.shape[0])
    p,g,c = optimize(w,b,X_train,Y_train,num,beta)
    w = p["w"]
    b = p["b"]
    Y_train_P = predict(w,b,X_train)
    Y_test_P = predict(w,b,X_test)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_P - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_P - Y_test)) * 100))
    ans = {"w":w,
           "b":b,
           "costs":c,
           "Y_train_P":Y_train_P,
           "Y_test_P":Y_test_P}
    return ans
ans = logistic(train_set_x_flatten,train_set_y,test_set_x_flatten,test_set_y)
# print(ans)