import numpy as np
import scipy.io as sio

# mnist data
data = sio.loadmat('hw01_data/mnist/train.mat')
data_mnist = np.array(data['trainX'])
# shuffle
np.random.shuffle(data_mnist)
# validation data
valid_data_mnist = data_mnist[:10000]
valid_label_mnist = valid_data_mnist[:,-1]
valid_data_mnist = valid_data_mnist[:,:-1]
# training data
train_data_mnist = data_mnist[10000:]
train_label_mnist = train_data_mnist[:,-1]
train_data_mnist = train_data_mnist[:,:-1]
# test data
test_data = sio.loadmat('hw01_data/mnist/test.mat')
test_data_mnist = test_data['testX']
# save data
np.savetxt('mnist_train_data.txt', train_data_mnist, fmt='%d')
np.savetxt('mnist_train_label.txt', train_label_mnist, fmt='%d')
np.savetxt('mnist_valid_data.txt', valid_data_mnist, fmt='%d')
np.savetxt('mnist_valid_label.txt', valid_label_mnist, fmt='%d')
np.savetxt('mnist_test_data.txt', test_data_mnist, fmt='%d')


# spam data

data = sio.loadmat('hw01_data/spam/spam_data.mat')
train_data_spam = np.array(data['training_data'])
train_label_spam = np.array(data['training_labels'])
data_spam = np.concatenate((train_data_spam, train_label_spam.T), axis=1)

np.random.shuffle(data_spam)
valid_data_spam = data_spam[:1034]
valid_label_spam = valid_data_spam[:,-1]
valid_data_spam = valid_data_spam[:,:-1]

train_data_spam = data_spam[1034:]
train_label_spam = train_data_spam[:,-1]
train_data_spam = train_data_spam[:,:-1]

test_data_spam = np.array(data['test_data'])

np.savetxt('spam_train_data.txt', train_data_spam, fmt='%d')
np.savetxt('spam_train_label.txt', train_label_spam, fmt='%d')
np.savetxt('spam_valid_data.txt', valid_data_spam, fmt='%d')
np.savetxt('spam_valid_label.txt', valid_label_spam, fmt='%d')
np.savetxt('spam_test_data.txt', test_data_spam, fmt='%d')




# cifar data
data = sio.loadmat('hw01_data/cifar/train.mat')
data_cifar = np.array(data['trainX'])
np.random.shuffle(data_cifar)
valid_data_cifar = data_cifar[:5000]
valid_label_cifar = valid_data_cifar[:, -1]
valid_data_cifar = valid_data_cifar[:, :-1]

train_data_cifar = data_cifar[5000:]
train_label_cifar = train_data_cifar[:, -1]
train_data_cifar = train_data_cifar[:, :-1]

test_data = sio.loadmat('hw01_data/cifar/test.mat')
test_data_cifar = test_data['testX']
np.savetxt('cifar_train_data.txt', train_data_cifar, fmt='%d')
np.savetxt('cifar_train_label.txt', train_label_cifar, fmt='%d')
np.savetxt('cifar_valid_data.txt', valid_data_cifar, fmt='%d')
np.savetxt('cifar_valid_label.txt', valid_label_cifar, fmt='%d')
np.savetxt('cifar_test_data.txt', test_data_cifar, fmt='%d')
