import numpy as np

# Values from previous output
layer_outputs = [4.8, 1.21, 2.385]

# Check out formula for Softmax on page 98

# e - mathematical constant, we use E here to match a common coding
# style where constants are Uppercased.
E = 2.71828182846

# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)

# Now normalize values
norm_values = exp_values / np.sum(exp_values)
print('Normalized exponentited values:')
print(norm_values)

print('Sum of normalized values:', sum(norm_values))
