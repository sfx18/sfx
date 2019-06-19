import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m


def func(x):
    return 1 / (1 + m.exp(-x))

DD = pd.read_csv("haberman.csv")

x = DD.values[:,:2]
y = DD.values[:,3:]-1

x = np.array(x)
y = np.array(y)


x = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)

W = np.zeros(3)

l = y.shape[0]

n = 0.01

for i in range(200):
    for j in range(3):
      sum=0
      for k in range(l):
          sum += x[k,j] * y[k] *func(-y[k] * W.dot(x[k]))
      W[j] = W[j] + n * (1 / l) * sum

predskazanie = x.dot(W)

for i in range(l):
  sum += 1 if predskazanie[i] == y[i] else -1
sum*=-1
print('Tochnost = ',sum/l)

red = []
for i in range(y.shape[0]):
    if y[i] > 0:
        red.append([x[i,1],x[i,2]])

blue = []
for i in range(y.shape[0]):
    if y[i] == 0:
        blue.append([x[i,1],x[i,2]])

red = np.array(red)
blue = np.array(blue)

x2 = (-W[0] - W[1] * x) / W[2]

plt.scatter(red[:,0], red[:,1], color="red")
plt.scatter(blue[:,0], blue[:,1], color="blue")
plt.plot(x, x2, color="green")
plt.show()