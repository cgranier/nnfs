import numpy as np

a = [1, 2, 3]
print(np.expand_dims(np.array(a), axis=0))

b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T

print(a)
print(b)

print(np.dot(a,b))