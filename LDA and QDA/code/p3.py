import numpy as np
import matplotlib.pyplot as plt

# problem 3

X1 = []
X2 = []
for i in range(100):
    x1 = np.random.normal(3,3)
    x2 = x1/2 + np.random.normal(4,2)
    X1.append(x1)
    X2.append(x2)
sample = np.array([X1,X2])

mean = np.mean(sample,axis=1)
print(mean)
cov = np.cov(sample)
print(cov)
w,v = np.linalg.eig(cov)



v1 = v[:,0]
v1 = v1 / np.linalg.norm(v1) * w[0]

v2 = v[:,1]
v2 = v2 / np.linalg.norm(v2) * w[1]

print(w)
print(v)

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
ax.set_xticks(np.arange(-15,16,1))
ax.set_yticks(np.arange(-15,16,1))
plt.grid()
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.scatter(X1,X2)
ax =plt.axes()
ax.arrow(mean[0], mean[1], v1[0], v1[1],head_length=1,head_width=0.5)
ax.arrow(mean[0], mean[1], v2[0], v2[1],head_length=1,head_width=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('100 samples and eigenvectors of the covariance matrix')
plt.savefig('p3_1.png')
plt.show()


for i in range(100):
    sample[:,i] = np.array(v.T * np.matrix(sample[:,i].reshape((2,1))- mean.reshape((2,1)))).reshape(2)

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
ax.set_xticks(np.arange(-15,16,1))
ax.set_yticks(np.arange(-15,16,1))
plt.grid()
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.scatter(sample[0,:], sample[1,:])
ax =plt.axes()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('100 rotated samples')
plt.savefig('p3_2.png')
plt.show()