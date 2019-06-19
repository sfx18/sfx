import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import math as m


def func(x):
    return 1 / (1 + m.exp(-x))

l = 100

iris = ds.load_iris()
x = iris.data[:l,1:3]
y = iris.target[:l] * 2 - 1

x = np.array(x)
y = np.array(y)

x = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)

W = np.zeros(3)

l = y.shape[0]

n = 0.01

for i in range(2000):
    for j in range(3):
      sum=0
      for k in range(l):
          sum += x[k,j] * y[k] *func(-y[k] * W.dot(x[k]))
      W[j] = W[j] + n * (1 / l) * sum

predskazanie = x.dot(W)

for i in range(l):
  sum += 1 if predskazanie[i] == y[i] else -1
sum*=-1
print('Точность = ',sum/l)

red = x[y > 0]
blue = x[y <= 0]

x2 = (-W[0] - W[1] * x) / W[2]

plt.scatter(red[:,1], red[:,2], color="red")
plt.scatter(blue[:,1], blue[:,2], color="blue")
plt.plot(x, x2, color="green")
plt.show()
