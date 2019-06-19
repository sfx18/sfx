import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds


DS = ds.load_boston()
X = DS.data
Y = DS.target

X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)

result = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

Y_predict = X.dot(result)

R2 = 1 - ((Y - Y_predict)**2).sum() / ((Y - Y.mean())**2).sum()
print("R2: ", R2)

plt.scatter(Y, Y_predict)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], c='r')
plt.show()
