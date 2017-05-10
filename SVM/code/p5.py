from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

def toCSV(array, filename):
    length = array.shape[0]
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id','Category'])
        for i in range(length):
            writer.writerow([i,int(array[i])])


# MNIST

data = sio.loadmat('hw01_data/mnist/train.mat')
data_mnist = np.array(data['trainX'])
print(data_mnist.shape)
np.random.shuffle(data_mnist)
data_mnist = data_mnist[:20000,:]
train_data_mnist = data_mnist[:,:-1]
train_label_mnist = data_mnist[:,-1]

test_data = sio.loadmat('hw01_data/mnist/test.mat')
test_data_mnist = test_data['testX']
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
scores = []
print('start...')
for i in range(1):
    clf = SVC(C = 0.000001, kernel='linear')
    score = cross_val_score(clf, train_data_mnist, train_label_mnist, cv=5, scoring='accuracy')
    scores.append(score)
    print(np.mean(score))
    clf.fit(train_data_mnist, train_label_mnist)
    predict_Y = clf.predict(test_data_mnist)
    toCSV(predict_Y, 'mnist_'+kernel[i]+'_result.csv')
print(scores)

# SPAM

data = sio.loadmat('hw01_data/spam/spam_data_2.mat')
train_data_spam = np.array(data['training_data'])
train_label_spam = np.array(data['training_labels'])
test_data_spam = np.array(data['test_data'])
data_spam = np.concatenate((train_data_spam, train_label_spam.T), axis=1)
np.random.shuffle(data_spam)
print(data_spam.shape)
train_data_spam = data_spam[:,:-1]
train_label_spam = data_spam[:,-1]
kernel = ['linear', 'rbf', 'poly', 'sigmoid']
scores = []
print('start...')
for i in range(1):
    clf = SVC(C=10, kernel=kernel[i])
    score = cross_val_score(clf, train_data_spam, train_label_spam, cv=5, scoring='accuracy')
    clf.fit(train_data_spam,train_label_spam)
    predict_Y = clf.predict(test_data_spam)
    toCSV(predict_Y, kernel[i]+'_result.csv')
    scores.append(score)
    print(np.mean(score))
print(scores)