import numpy as np
import matplotlib.pyplot as plt



def low_rank_approximation(D, k):
    U,s,V = np.linalg.svd(D)
    s1 = np.zeros((U.shape[0], V.shape[0]))
    s1[:k,:k] = np.diag(s[:k])
    return (U.dot(s1)).dot(V)


def MSE(X, X1):
    return np.sum((X-X1)**2)/X.shape[0]/X.shape[1]



img = plt.imread('hw7_data/low-rank_data/face.jpg')

img1 = low_rank_approximation(img, 18)
img2 = low_rank_approximation(img, 20)
img3 = low_rank_approximation(img, 100)

plt.imshow(img1, cmap='gray')
plt.title('rank 5')
plt.savefig('sky_rank-5.png')

plt.imshow(img2, cmap='gray')
plt.title('rank 20')
plt.savefig('sky_rank-20.png')

plt.imshow(img3, cmap='gray')
plt.title('rank 100')
plt.savefig('sky_rank-100.png')


plt.figure()
mse = []
for i in range(100):
    img_new = low_rank_approximation(img,i+1)
    mse.append(MSE(img,img_new))
plt.plot(mse)
plt.xlabel('rank')
plt.ylabel('MSE')
plt.title('MSE vs rank')
plt.savefig('sky_MSE.png')


img_sky = low_rank_approximation(img, 80)
plt.figure()
plt.subplot('121')
plt.title('d=80')
plt.imshow(img_sky, cmap='gray')

plt.subplot('122')
plt.title('orginal')
plt.imshow(img, cmap='gray')
plt.savefig('face_d=80.png')
plt.show()
