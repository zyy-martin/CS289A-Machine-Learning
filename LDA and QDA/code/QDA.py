import numpy as np
from sklearn import preprocessing

class QDA:
    sigma = None
    mu = None
    pi = None
    n_feature =0
    n_class = 0
    h = 0
    def __init__(self,c,f, h):
        self.n_class = c
        self.n_feature = f
        self.sigma = np.zeros((self.n_class+1, self.n_feature, self.n_feature))
        self.mu = np.zeros((self.n_class, self.n_feature))
        self.pi = np.zeros((self.n_class, 1))
        self.h = h
    def fit(self,X,y):
        X_norm = preprocessing.normalize(X)
        X_class = {}
        for i in range(y.shape[0]):
            if y[i] in X_class:
                X_class[y[i]].append(X_norm[i, :].tolist())
            else:
                X_class[y[i]] = []
                X_class[y[i]].append(X_norm[i, :].tolist())
        for key in X_class:
            X_class[key] = np.array(X_class[key])

        for i in range(self.n_class):
            self.pi[i] = X_class[i].shape[0] / X_norm.shape[0]
        for i in range(self.n_class):
            self.mu[i, :] = np.mean(X_class[i], axis=0)
            cov_matrix = np.cov(X_class[i].T)
            cov_matrix += np.identity(cov_matrix.shape[0]) * self.h
            self.sigma[i, :, :] = cov_matrix
        for i in range(self.n_class):
            self.sigma[self.n_class, :, :] += self.sigma[i, :, :] * X_class[i].shape[0] / X_norm.shape[0]

    def predict(self, X):
        X_norm = preprocessing.normalize(X)
        inv_cov = np.zeros((self.n_class,self.n_feature,self.n_feature))
        for i in range(self.n_class):
            inv_cov[i] = np.linalg.inv(np.matrix(self.sigma[i]))


        const =  np.zeros((self.n_class,1))
        for i in range(self.n_class):
            const[i] = 0.5 * np.matrix(self.mu[i]) * inv_cov[i] * np.matrix(self.mu[i]).T

        const1 = np.zeros((self.n_class,self.n_feature))
        for i in range(self.n_class):
            const1[i] = np.matrix(self.mu[i]) * inv_cov[i]

        logdet = np.zeros((self.n_class,1))
        for i in range(self.n_class):
            sign, logdet[i] = np.linalg.slogdet(np.matrix(self.sigma[i]))



        predict_res = np.zeros((X.shape[0],1))
        for i in range(X_norm.shape[0]):
            index = 0
            val = const1[0] * np.matrix(X_norm[i]).T - const[0] -0.5 * np.matrix(X_norm[i]) * inv_cov[0] * np.matrix(X_norm[i]).T + np.log(self.pi[0]) - 0.5* logdet[0]
            for j in range(self.n_class-1):
                temp_val = const1[j+1] * np.matrix(X_norm[i]).T - const[j+1] -0.5 * np.matrix(X_norm[i]) * inv_cov[j+1] * np.matrix(X_norm[i]).T + np.log(self.pi[j+1])- 0.5* logdet[j+1]
                if temp_val > val:
                    val = temp_val
                    index = j + 1
            predict_res[i] = index

        return predict_res










