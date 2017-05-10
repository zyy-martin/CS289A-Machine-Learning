import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# problem 2


delta = 0.01
x = np.arange(-5.0, 5.0, delta)
y = np.arange(-5.0, 5.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 2.0, 1.0, 1.0, 1.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 2.0, 2.0, -1.0, -1.0, 1.0)
Z = Z1 - Z2


plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('problem 2e')
plt.savefig('2e.png')
plt.show()