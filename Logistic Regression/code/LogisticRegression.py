import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import random


def loss(z, y):
    # if z == 1.0:
    #     z =0.999999
    # if z == 0.0:
    #     z = 0.00001
    if int(y) == 0:
        return - np.log(1 - z)
    if int(y) == 1:
        return - np.log(z)
    else:
        print('warning')
        return - y * np.log(z) - (1 - y) * np.log(1 - z)

def cost(X,y,w,lam):
    J = 0
    n = y.shape[0]
    for i in range(X.shape[0]):
        J += loss(scipy.special.expit(np.dot(X[i, :], w)), y[i])
        # J /= n
        J += lam * np.linalg.norm(w) ** 2
    return J

class logistic_regression_l2:
    def __init__(self, lam, epsilon, n_iteration):
        self.lam = lam
        self.n_iteration = n_iteration
        self.epsilon = epsilon
        self.w = None

    def train(self, X, y, method='batch'):
        if method == 'batch':
            w = np.zeros((X.shape[1],1))
            loss_fun = []
            for i in range(self.n_iteration):
                w += self.epsilon * (np.matrix(X).T * (y - scipy.special.expit(np.dot(X, w))) - 2 * self.lam * w )
            #     J = cost(X,y,w,self.lam)
            #     loss_fun.append(J)
            # plt.plot(loss_fun)
            # plt.xlabel('number of iterations')
            # plt.ylabel('cost function')
            # plt.title('cost function vs number of iterations: batch')
            # plt.savefig('1.png')
            # plt.show()
            self.w = w
        if method == 'SGD':
            w = np.zeros((X.shape[1], 1))
            n = X.shape[0]
            loss_fun = []
            for i in range(self.n_iteration):
                index = random.randint(0, n-1)
                delta = self.epsilon * ((y[index] - scipy.special.expit(np.dot(X[index, :].reshape((1, X.shape[1])), w))) * X[index, :].reshape((X.shape[1],1)) - 2 * self.lam * w)
                w += delta
            #     J = cost(X, y, w, self.lam)
            #     loss_fun.append(J)
            # plt.plot(loss_fun)
            # plt.xlabel('number of iterations')
            # plt.ylabel('cost function')
            # plt.title('cost function vs number of iterations: SGD')
            # plt.savefig('2_1.png')
            # plt.show()
            self.w = w
        if method == 'SGD_vstep':
            w = np.zeros((X.shape[1], 1))
            n = X.shape[0]
            loss_fun = []
            for i in range(self.n_iteration):
                index = random.randint(0, n - 1)
                delta = self.epsilon / (i+1)  * (
                (y[index] - scipy.special.expit(np.dot(X[index, :].reshape((1, X.shape[1])), w))) * X[index, :].reshape(
                    (X.shape[1], 1)) - 2 * self.lam * w)
                w += delta
            #     J = cost(X, y, w, self.lam)
            #     loss_fun.append(J)
            # plt.plot(loss_fun)
            # plt.xlabel('number of iterations')
            # plt.ylabel('cost function')
            # plt.title('cost function vs number of iterations: SGD with changing stepsize')
            # plt.savefig('3_1.png')
            # plt.show()
            self.w = w



    def predict(self, X):
        res = scipy.special.expit(np.dot(X, self.w))
        res[res > 0.5] = 1
        res[res <= 0.5] = 0
        return res

