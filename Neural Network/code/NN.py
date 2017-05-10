import numpy as np


class NeuralNet:
    def __init__(self, in_size, hidden_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.V = np.random.normal(0, 0.05, (self.hidden_size, self.in_size))
        self.W = np.random.normal(0, 0.2, (self.out_size, self.hidden_size))
        self.b1 = 0.01 * np.random.uniform(size=(self.hidden_size,1))
        self.b2 = 0.01 * np.random.uniform(size=(self.out_size,1))

    def train(self, X, y, eps=0.001, max_iter=10000, batch_size = 100, reg = 0.001):
        N, _ = X.shape
        y_one_hot = np.zeros((N, self.out_size))

        for i in range(N):
            y_one_hot[i, int(y[i])] = 1

        loss = []
        train_acc = []
        count = 0
        n_batch = int(N / batch_size)

        while count < max_iter:
            count += 1
            rand_ind = np.random.choice(np.arange(N), batch_size)
            cur_X = X[rand_ind, :].T
            cur_y = y_one_hot[rand_ind, :].T

            # forward prop

            h = np.tanh(np.dot(self.V, cur_X) + self.b1)
            z =self.sigmoid(np.dot(self.W, h) + self.b2)

            # decay learning rate every 2 epoch2
            if count % batch_size == 1:
                predicted_Y = self.predict(X)
                acc = self.accuracy(predicted_Y, y)
                print(acc)
                train_acc.append(acc)
                l = np.sum(-cur_y * np.log(z) - (1 - cur_y) * np.log(1 - z))
                loss.append(l)
            if count % (2 * n_batch) == 0:
                print('another two epoch')
                eps *= 0.9




            dv = ((self.W.T.dot(z - cur_y)) * (1 - h ** 2)).dot(cur_X.T) + reg * self.V
            dw = np.dot(z - cur_y, h.T) + reg * self.W
            db1 = np.sum(((self.W.T.dot(z - cur_y)) * (1 - h ** 2)), axis=1, keepdims=True)
            db2 = np.sum(z - cur_y,axis=1, keepdims=True)


            self.W -= eps * dw / batch_size
            self.V -= eps * dv / batch_size
            self.b1 -= eps * db1 / batch_size
            self.b2 -= eps * db2 / batch_size
        print(loss)
        return loss, train_acc


    def predict(self, X):

        x = X.T
        h = np.tanh(np.dot(self.V, x) + self.b1)
        z = self.sigmoid(np.dot(self.W, h) + self.b2)
        label = np.argmax(z, axis=0)
        return label


    def accuracy(self, y_predicted, y_true):
        score = 0
        for i in range(y_true.shape[0]):
            if int(y_predicted[i]) == int(y_true[i]):
                score += 1
        return score / y_true.shape[0]

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

































