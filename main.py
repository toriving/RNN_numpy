import numpy as np
from RNN_numpy import RNN

a = RNN(6)

x_train = np.array([[1,2,3,4,5]])
y_train = np.array([[2,3,4,5,1]])

a.train(x_train,y_train, learning_rate=0.01, epoch = 200)

print(a.predict(np.array([2,1,3])))
print(a.predict(np.array([2,1])))

print(a.predict(np.array([1,2,3,4,5])))
print(a.predict(np.array([2,3,4,5,1])))
print(a.predict(np.array([3,4,5,1,2])))
print(a.predict(np.array([4,5,1,2,3])))
print(a.predict(np.array([5,1,2,3,4])))
print(a.predict(np.array([3])))


a1 = a.predict(np.array([1]))
a2 = a.predict(a1)
a3 = a.predict(a2)
a4 = a.predict(a3)
a5 = a.predict(a4)
print(a1,a2,a3,a4,a5)

