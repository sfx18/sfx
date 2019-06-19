import numpy as np
import matplotlib.pyplot as plt


def F(x, b, k):
    result = 0
    for i in range(k):
        result += (x ** i) * b[i]
    return result


k = 8

x = np.sort(5 * np.random.rand(40, 1), axis=0)
y = (np.sin(x)*np.cos(x)**2)

x = np.array(x)
y = np.array(y)

mem = {}
a = []
for i in range(k):
    a.append([])
    for j in range(k):
        if not mem.get(i + j):
            mem[i + j] = (x ** (i + j)).mean()
        a[i].append(mem[i + j])

a = np.array(a)

s = []
for i in range(k):
    s.append((x ** i * y).mean())

s = np.array(s)

b = np.linalg.solve(a, s)

y_predict = F(x, b, k)

plt.scatter(x, y, color='r')
plt.plot(x, y_predict, color="b")
plt.show()
