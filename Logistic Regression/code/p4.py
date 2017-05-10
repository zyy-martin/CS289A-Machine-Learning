import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import LogisticRegression
import csv


data = sio.loadmat('data.mat')
X_test = data['X_test']
X = data['X']

X = np.concatenate((X, np.ones((X.shape[0],1))),axis=1)
X_test = np.concatenate((X_test, np.ones((X_test.shape[0],1))),axis=1)
X = preprocessing.normalize(X)
X_test = preprocessing.normalize(X_test)
y = data['y']
X_train = X[:5000, :]
y_train = y[:5000, :]
X_val = X[5000:, :]
y_val = y[5000:, :]


# batch
clf = LogisticRegression.logistic_regression_l2(0.0001, 0.001, 500)
clf.train(X_train, y_train,method='batch')
#
# SGD
clf = LogisticRegression.logistic_regression_l2(0.0001, 0.1, 500)
clf.train(X_train, y_train,method='SGD')

# SGD with changing step size
clf = LogisticRegression.logistic_regression_l2(0.0001, 50, 100)
clf.train(X_train, y_train,method='SGD_vstep')




# res = clf.predict(X_val)
# error = 0
# for i in range(res.shape[0]):
#     if int(res[i]) != int(y_val[i]):
#         error += 1
# print(error/res.shape[0])



# Kaggle
clf = LogisticRegression.logistic_regression_l2(0.00000, 0.5, 200000)
clf.train(X, y, method='batch')
res = clf.predict(X_test)
with open('result.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Id','Category'])
    for i in range(res.shape[0]):
        writer.writerow([str(i), str(int(res[i]))])
