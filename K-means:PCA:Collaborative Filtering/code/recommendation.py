import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt

def low_rank_approximation(D, k):
    U,s,V = np.linalg.svd(D, full_matrices=False)
    s1 = np.zeros((U.shape[1], k))
    s2 = np.zeros((k, V.shape[0]))
    s1[:k,:k] = np.diag(np.power(s[:k], 0.5))
    s2[:k, :k] = np.diag(np.power(s[:k], 0.5))
    return U.dot(s1), s2.dot(V)

def MSE(train_data, U, V, non_nan_set):
    mse = 0
    for item in non_nan_set:
        i = item[0]
        j = item[1]
        mse += (U[i,:].dot(V[:,j])-train_data[i,j])**2
    return mse

def predict(u, v, i, j):
    rating = u[i,:].dot(v[:,j])
    if rating >=0 :
        return 1
    else:
        return 0
# validation set
valid = []
with open('hw7_data/joke_data/validation.txt') as f:
    for line in f:
        line = line.strip().split(',')
        valid.append(line)

# training data
data = scipy.io.loadmat('hw7_data/joke_data/joke_train.mat')
train_data = data['train']

# non-NaN set
non_nan_set = []
for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        if math.isnan(train_data[i,j]):
            continue
        non_nan_set.append([i,j])

train_data_zero = np.nan_to_num(train_data)

dimension =[2,5,10,20]
loss = []
error = []
for d in dimension:
    u, v = low_rank_approximation(train_data_zero, d)
    print(u.shape, v.shape)
    loss.append(MSE(train_data, u, v, non_nan_set))
    error_d = 0
    print(d)
    for line in valid:
        predicted_r = predict(u, v, int(line[0])-1,int(line[1])-1)
        if predicted_r != int(line[2]):
            error_d += 1
    error_d /= len(valid)
    error.append(1-error_d)
plt.figure()
plt.plot(dimension, loss)
plt.xlabel('d')
plt.ylabel('MSE')
plt.title('MSE vs d')
plt.savefig('loss_d.png')

plt.figure()
plt.plot(dimension, error)
plt.xlabel('d')
plt.ylabel('validation accuracy')
plt.title('validation accuracy vs d')
plt.savefig('accu_d.png')






# a = np.array([[1,2,3],[2,2,5],[2,9,6],[1,2,0]])
# print(low_rank_approximation(a,3))