import numpy as np
from data_split import preprocess_data
from LDA import LDA
from QDA import QDA
import scipy.io as sio
import csv


# problem 6d

data = sio.loadmat('spam/spam_data_2.mat')
train_X = data['training_data']
train_y = data['training_labels'][0]
test_X = data['test_data']
# validate_X = train_X[:5000,:]
# validate_y = train_y[:5000]
# train_X = train_X[5000:,:]
# train_y = train_y[5000:]



cls_lda = LDA(2, 104,0.0001)
cls_lda.fit(train_X,train_y)
y_predict = cls_lda.predict(test_X)


# error = 0
# for i in range(validate_X.shape[0]):
#     error += (int(validate_y[i]) != int(y_predict[i]))
# error /= validate_X.shape[0]
# print(error)

with open('spam_predict.csv','wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Id','Category'])
    for i in range(y_predict.shape[0]):
        writer.writerow([i, int(y_predict[i][0])])