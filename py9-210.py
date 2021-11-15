import numpy as np

# Example layer output
z = np.array([[1, 2, -3, -4],
             [2, -7, -1, 3],
             [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# ReLU activation's derivative
drelu = np.zeros_like(z)
print(drelu)

drelu[z > 0] = 1

print(drelu)

# The chain rule
drelu *= dvalues

print(drelu)

print(dvalues[z < 0])

print(len(z))

print(len(z[0]))

print(z.shape)

print(z.shape[0])