from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


# minst
train_X = np.loadtxt('mnist_train_data.txt', dtype=int)
train_Y = np.loadtxt('mnist_train_label.txt', dtype=int)
valid_X = np.loadtxt('mnist_valid_data.txt', dtype=int)
valid_Y = np.loadtxt('mnist_valid_label.txt', dtype=int)

print(train_X.shape, train_Y.shape)

num = [100, 200, 500, 1000, 2000, 5000, 10000]
err_valid = []
err_train = []
for i in range(7):
    clf = SVC(C=1, kernel='linear')
    clf.fit(train_X[:num[i],:], train_Y[:num[i]])
    predict_train_Y = clf.predict(train_X[:num[i],:])
    predict_Y = clf.predict(valid_X)
    err_valid.append(1 - accuracy_score(valid_Y, predict_Y))
    err_train.append(1 - accuracy_score(train_Y[:num[i]], predict_train_Y))
    print(i)
plt.plot(num, err_valid, 'rs', label='validation error' )
plt.plot(num, err_train, 'bs', label='training error')
plt.legend()
plt.xlabel('# training samples')
plt.ylabel('error')
plt.title('training errors & validation errors vs # training samples, MNIST')
plt.savefig('p2-MNIST.png')
plt.show()

# spam

train_X = np.loadtxt('spam_train_data.txt', dtype=int)
train_Y = np.loadtxt('spam_train_label.txt', dtype=int)
valid_X = np.loadtxt('spam_valid_data.txt', dtype=int)
valid_Y = np.loadtxt('spam_valid_label.txt', dtype=int)

train_size = train_X.shape[0]

print(train_size)

num = [100, 200, 500, 1000, 2000, train_size]
err_valid = []
err_train = []
for i in range(6):
    clf = SVC(C=1, kernel='linear')
    clf.fit(train_X[:num[i],:], train_Y[:num[i]])
    predict_train_Y = clf.predict(train_X[:num[i],:])
    predict_Y = clf.predict(valid_X)
    err_valid.append(1 - accuracy_score(valid_Y, predict_Y))
    err_train.append(1 - accuracy_score(train_Y[:num[i]], predict_train_Y))
    print(i)
plt.plot(num, err_valid, 'rs', label='validation error' )
plt.plot(num, err_train, 'bs', label='training error')
plt.legend()
plt.xlabel('# training samples')
plt.ylabel('error')
plt.title('training errors & validation errors vs # training samples, SPAM')
plt.savefig('p2-SPAM.png')
plt.show()

# cifar
#
train_X = np.loadtxt('cifar_train_data.txt', dtype=int)
train_Y = np.loadtxt('cifar_train_label.txt', dtype=int)
valid_X = np.loadtxt('cifar_valid_data.txt', dtype=int)
valid_Y = np.loadtxt('cifar_valid_label.txt', dtype=int)

train_size = train_X.shape[0]

print(train_size)

num = [100, 200, 500, 1000, 2000, 5000]
err_valid = []
err_train = []
for i in range(6):
    clf = SVC(C=1, kernel='linear')
    clf.fit(train_X[:num[i],:], train_Y[:num[i]])
    predict_train_Y = clf.predict(train_X[:num[i],:])
    predict_Y = clf.predict(valid_X)
    err_valid.append(1 - accuracy_score(valid_Y, predict_Y))
    err_train.append(1 - accuracy_score(train_Y[:num[i]], predict_train_Y))
    print(i)
plt.plot(num, err_valid, 'rs', label='validation error' )
plt.plot(num, err_train, 'bs', label='training error')
plt.legend()
plt.xlabel('# training samples')
plt.ylabel('error')
plt.title('training errors & validation errors vs # training samples, CIFAR')
plt.savefig('p2-CIFAR.png')
plt.show()