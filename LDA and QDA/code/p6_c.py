import numpy as np
from data_split import preprocess_data
from LDA import LDA
from QDA import QDA
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis
import csv

# problem 6ci, 6cii


num = [100,200,500,1000,2000,5000,10000,30000,50000]


error_LDA =[]
error_QDA =[]
for i in range(9):
    preprocess_data('1', 10, num[i])
    data = sio.loadmat('data_class_1.mat')
    X_train = data['X_train']
    y_train = data['y_train'][0]
    X_validate = data['X_validate']
    y_validate = data['y_validate'][0]

    sk_cls = discriminant_analysis.QuadraticDiscriminantAnalysis()
    sk_cls.fit(X_train,y_train)
    y_predict_1 = sk_cls.predict(X_validate)


    cls = LDA(10, 784, 0.0001)
    cls.fit(X_train,y_train)
    y_predict = cls.predict(X_validate)


    v_error = 0
    for j in range(10000):
        v_error += (int(y_validate[j]) != int(y_predict[j]))
    v_error /= 10000
    error_LDA.append(v_error)
    print(v_error)


    cls = QDA(10,784,0.0001)
    cls.fit(X_train,y_train)
    y_predict = cls.predict(X_validate)
    v_error = 0
    for j in range(10000):
        v_error += (int(y_validate[j]) != int(y_predict[j]))
    v_error /= 10000
    error_QDA.append(v_error)
    print(v_error)

plt.plot(num, error_LDA,'rs')
plt.xlabel('# training samples')
plt.ylabel('error')
plt.title('LDA validation error vs # training samples')
plt.savefig('LDA_mnist_3.png')
plt.show()

plt.plot(num, error_QDA,'rs')
plt.xlabel('# training samples')
plt.ylabel('error')
plt.title('QDA validation error vs # training samples')
plt.savefig('QDA_mnist_3.png')
plt.show()



# problem 6civ, kaggle


data = sio.loadmat('hw3_mnist_dist/train.mat')
train_X = data['trainX']
train_y = train_X[:,-1]
train_X = train_X[:,:-1]
data = sio.loadmat('hw3_mnist_dist/test.mat')
test_X = data['testX']
cls = QDA(10,784,0.0001)
cls.fit(train_X,train_y)
predict = cls.predict(test_X)
with open('mnist_predict_QDA.csv','wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Id','Category'])
    for i in range(predict.shape[0]):
        writer.writerow([i, int(predict[i][0])])
