import numpy as np
import scipy.io
import matplotlib.pyplot as plt


class kmeans:
    def __init__(self, n_clusters =5):
        self.n_clusters = n_clusters


    def fit(self, X):
        y = np.random.randint(self.n_clusters,size=X.shape[0])
        means = self.calculate_mean(X, y)
        while True:
            old_y  = y
            y = self.update_y(X, means)
            means = self.calculate_mean(X, y)
            print(y)
            print(means)
            if np.linalg.norm(old_y - y) < 1e-10:
                break
        return y, means

    def calculate_mean(self, X, y):
        means = np.zeros((self.n_clusters, X.shape[1]))
        cluster_size = np.zeros(self.n_clusters)
        for i in range(X.shape[0]):
            means[int(y[i])] += X[i]
            cluster_size[int(y[i])] += 1
        means /= cluster_size[:,None]
        return means

    def update_y(self, X, means):
        y = np.zeros((X.shape[0],),dtype=np.int)
        for i in range(X.shape[0]):
            y[i] = self.cloesest_cluster(X[i,:], means)
        return y

    def cloesest_cluster(self, x, means):
        label = -1
        distance = np.inf
        for i in range(self.n_clusters):
            new_dist =np.linalg.norm(x - means[i,:])
            if new_dist < distance:
                distance = new_dist
                label = i
        return label



    

data = scipy.io.loadmat('hw7_data/mnist_data/images.mat')
img = data['images']
new_img = np.zeros((60000, 784))
for i in range(60000):
    for j in range(28):
        new_img[i,j*28:(j+1)*28] = img[j,:,i]

km = kmeans(15)
y, means = km.fit(new_img)
for i in range(15):
    plt.imshow(means[i,:].reshape((28,28)))
    plt.savefig('15_clusters_'+str(i)+'.png')





