
import numpy as np
import scipy.io as sio
import numpy.linalg as li
import scipy.optimize as so
import matplotlib.pyplot as plt


#Объявление матриц

A = np.array([[1, 2, 3], [-2,3,0],[5,1,4]])
print (str(A) + ' = A\n')
B = np.array([[1, 2, 3], [2,4,6],[3,7,2]])
print (str(B) + ' = B\n')
u = np.array([[-4], [1], [1]])
print (str(u) + ' = u\n')
v = np.array([[3], [2], [10]])
print (str(v) + ' = v\n')

#Объявление случайной матрицы и вектора

C = np.random.random((100, 100))
print (str(C) + ' = C\n')
w = np.random.random((100, 1))
print (str(w) + ' = w\n')

#Вычисление выражений

print (str(A + B) + ' = A + B\n')
print (str(A * B) + ' = AB\n')
print (str(A * A * A) + ' = A^3\n')
print (str(A * B * A) + ' = ABA\n')
print (str(v.T * A.T * (u + 2*v)) + ' = (v^T)(A^T)(u+2v)\n')
print (str(u * v) + ' = uv\n')
print (str(C * w) + ' = Cw\n')
print (str(w.T * C) + ' = (w^T)C\n')

#матрица,элементы которой i*j

D = np.fromfunction(lambda i,j : i*j, (20, 20))
print(str(D).replace('.','').replace('  ',' '))

#сохранение матрицы в файл

sio.savemat ('data.mat', {'A': A})
A1 = sio.loadmat('data.mat')['A']
print (A1)

#сумма положительных эл-ов матриц

print (str(A[A>0].sum()) + ' = Summa A[A>0]\n')
print (str(B[B>0].sum()) + ' = Summa B[B>0]\n')
print (str(A[A>0].sum()+B[B>0].sum()) + ' = Summa A+B[A,B>0]\n')

#записать матрицу в строку 

AA = np.reshape(A, (1, 9))
print(AA)
print(str(AA[:,1::2]) + ' = chetnie pozicii\n')

#найти обратные и псевдообратные матрицы

print (str(li.inv(A)) + ' = obratnaj A\n')
print (str(li.pinv(B)) + ' = psevdoobratnaj A\n')
print (str(li.inv(C)) + ' = obratnaj C\n')
print (str(li.inv(A).dot(A)) + ' = edenichnaj A\n')
print (str(li.pinv(B).dot(B)) + ' = edenichnaj B\n')
print (str(li.inv(C).dot(C)) + ' = edenichnaj C\n')

#решить систему ур-й

H = np.array([[32, 7, -6], [-5, -20, 3], [0, 1, -3]])
J = np.array([12, 3, 7])
k = np.linalg.solve(H,J)
print(k,'= x1, x2, x3')
lj = k*H
print('proverka')
print(lj[0, :].sum())
print(lj[1, :].sum())
print(lj[2, :].sum())

#нахождение собственных значений

print (li.eig(A))
print (li.eig(B))

#найти минимумы ф-й

def f1(x):
    return 5*(x-2)**4 - 1/(x**2+8)
x_min = so.minimize(f1, 0.0, method='BFGS')
print ('f1(x) = 5(x-2)^4 - 1/(x^2+8)\n' + str(x_min) + '\n')

def f2(x):
    return 4*(x[0]-3*x[1])**2 + 7*x[0]**4
def df2(x):
    df2dx1 = 8*(x[0]-3*x[1]) + 28*x[0]
    df2dx2 = 8*(x[0]-3*x[1])
    return np.array([df2dx1, df2dx2])
x_min = so.minimize(f2, [0.0, 0.0], method='BFGS', jac=df2)
print ('f2(x1,x2) = 4(x1-3*x2)^2 + 7*x1^4\n' + str(x_min))

#построить графики ф-й

def g1(x):
    return x**5 - 2*x**4 + 3*x - 7
def g2(x):
    return x**5 + 2*x**4 - 3*x - 7

x = np.linspace(-5, 5, 50)
g1y = g1(x)
g2y = g2(x)

plt.plot(x, g1y)
plt.plot(x, g2y)

plt.xlabel('x')
plt.ylabel('g1(x), g2(x)')
plt.legend(['g1', 'g2'])
plt.title('Plot')

plt.show()

#построить графики ф-й в двух системах координат

def g1(x):
    return x**5 - 2*x**4 + 3*x - 7
def g2(x):
    return x**5 + 2*x**4 - 3*x - 7

plt.subplot(1, 3, 1)
plt.plot(x, g1y)
plt.xlabel('x')
plt.ylabel('g1(x)')
plt.legend(['g1'])
plt.title('Plot #1')

plt.subplot(1, 3, 3)
plt.plot(x, g2y)
plt.xlabel('x')
plt.ylabel('g2(x)')
plt.legend(['g2'])
plt.title('Plot #2')

plt.show()

#найти по одному из нулей каждой ф-ии

g1x0 = so.brentq(g1, -5, 5)
g2x0 = so.brentq(g2, -5, 5)

print(g1x0)
print(g2x0)
