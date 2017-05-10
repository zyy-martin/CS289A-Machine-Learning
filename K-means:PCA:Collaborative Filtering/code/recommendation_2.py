import numpy as np
import scipy.io
import math
import csv
import matplotlib.pyplot as plt
from scipy import sparse

# training data
data = scipy.io.loadmat('hw7_data/joke_data/joke_train.mat')
train_data = data['train']


# validation set
valid = []
with open('hw7_data/joke_data/validation.txt') as f:
    for line in f:
        line = line.strip().split(',')
        valid.append(line)


# non-NaN set
non_nan_set = []
for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        if math.isnan(train_data[i,j]):
            continue
        non_nan_set.append([i,j])

train_data_zero = np.nan_to_num(train_data)



indicator = np.ones(train_data.shape)
for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        if math.isnan(train_data[i,j]):
            indicator[i, j] = 0
            train_data[i, j] = 0


def solve_u(u, v, data, indicator, l):
    u_new = np.zeros(u.shape)
    for user in range(u.shape[0]):
        data_user = data[user, :]
        indicator_user = indicator[user, :]
        #  indicator_user = np.diag(indicator_user)
        v_i = v[:, indicator_user.astype(int) == 1]
        data_user = data_user[indicator_user.astype(int) == 1]
        u_user = np.dot(data_user, v_i.T)

        u_user = np.dot(u_user, np.linalg.inv(l * np.identity(u.shape[1]) + v_i.dot(v_i.T)))
        u_new[user, :] = u_user
    return u_new


def solve_v(u, v, data, indicator, l):
    v_new = np.zeros(v.shape)
    for review in range(v.shape[1]):
        data_review = data[:, review]
        indicator_review = indicator[:, review]
        # indicator_review = np.diag(indicator_review)
        u_j = u[indicator_review.astype(int) == 1, :]
        data_review = data_review[indicator_review.astype(int) == 1]
        v_review = np.dot(u_j.T, data_review)
        v_review = np.dot(np.linalg.inv(np.dot(u_j.T, u_j) + l * np.identity(v.shape[0])), v_review)
        v_new[:, review] = v_review
    return v_new

def latent_indexing(data, d, indicator, max_step= 20, l = 300):
    u = np.random.uniform(-1, 1, (data.shape[0], d))
    v = np.random.uniform(-1, 1, (d, data.shape[1]))
    for i in range(max_step):
        u = solve_u(u, v, data, indicator, l)
        # print(MSE(train_data, u, v, non_nan_set))
        v = solve_v(u, v, data, indicator, l)
        # print(MSE(train_data, u, v, non_nan_set))
        print(i)
    return u, v

def predict(u, v, i, j):
    rating = u[i,:].dot(v[:,j])
    if rating > 0 :
        return 1
    else:
        return 0

def MSE(train_data, U, V, non_nan_set):
    mse = 0
    for item in non_nan_set:
        i = item[0]
        j = item[1]
        mse += (U[i,:].dot(V[:,j])-train_data[i,j])**2
    return mse


mses = []
accu = []
for d in range(20):
    u, v = latent_indexing(train_data, d, indicator)
    error_d = 0
    for line in valid:
        predicted_r = predict(u, v, int(line[0]) - 1, int(line[1]) - 1)
        if predicted_r != int(line[2]):
            error_d += 1
    error_d /= len(valid)
    accu.append(1 - error_d)
    mses.append(MSE(train_data, u, v, non_nan_set))

plt.figure()
plt.plot(range(20), mses)
plt.xlabel('d')
plt.ylabel('MSE')
plt.title('MSE vs d')
plt.savefig('new_loss_d.png')

plt.figure()
plt.plot(range(20), accu)
plt.xlabel('d')
plt.ylabel('validation accuracy')
plt.title('validation accuracy vs d')
plt.savefig('new_accu_d.png')


query = []
with open('hw7_data/joke_data/query.txt') as f:
    for line in f:
        line = line.strip().split(',')
        query.append(line)
result = []

for item in query:
    i = int(item[1]) - 1
    j = int(item[2]) - 1
    predicted = predict(u, v, i, j)
    result.append(predicted)

with open('results.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Id','Category'])
    for i, line in enumerate(result):
        writer.writerow([str(i+1), str(line)])



