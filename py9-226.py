import numpy as np

# made up sample
softmax_output = [0.7, 0.1, 0.2]

# shape it as a list of samples
softmax_output = np.array(softmax_output).reshape(-1, 1)

print(softmax_output)

print(np.eye(softmax_output.shape[0]))

print(softmax_output * np.eye(softmax_output.shape[0]))

print(np.diagflat(softmax_output))

print(np.dot(softmax_output, softmax_output.T))

print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))

# Softmax activation
class Activation_Softmax:
    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

dvalues = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)
print(Activation_Softmax.backward(1, dvalues))