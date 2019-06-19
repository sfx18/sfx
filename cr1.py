import numpy as np
import numpy.linalg as li
import scipy.integrate as itg
import sympy as sp
import math as m
import matplotlib.pyplot as plt
import pandas as pd

DD = pd.read_csv("diet_data.csv")
del DD ['Date']
del DD ['change']
del DD ['cals_per_oz']

x = DD.values[:,:6]
y = DD.values[:,6:]

x = np.array(x)
x = np.array(y)

x = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)

ves = np.linalg.solve((x.T.dot(x)), x.T.dot(y))
print(ves)

prognoz = x.dot(ves)

R2 = 1 - ((y - prognoz)**2).sum() / (((y - prognoz.mean())**2).sum())
print(R2)

plt.scatter(y, prognoz)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.show()
