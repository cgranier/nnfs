import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from numpy.core.function_base import _logspace_dispatcher

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities, making inputs negative
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Common loss class
class Loss:
    # Calculate the data and regularization loses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):

    	# Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by zero
        # Clip both sides to not drag mean towards any value (one too-large or too-little value)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values, only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Create dataset
X, y = spiral_data(samples=100, classes=3)

print('X:', X)
print('y', y)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer)
activation2 = Activation_Softmax()

# Create loss functions
loss_function = Loss_CategoricalCrossentropy()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make forward pass through activation function
# It takes the output of the first Dense layer here
activation1.forward(dense1.output)

# Make a forward pass through the second Dense layer
# It takes outputs of activation function of first layer as input
dense2.forward(activation1.output)

# Make a forward pass through activation functions
# It takes the output of second Dense layer here
activation2.forward(dense2.output)

# Lets see output of the first few samples:
print(activation2.output[:5])

# Perform a forward pass through activation function
# It takes the output of the second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)

# Print loss value
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(activation2.output, axis=1)

print('act2.out:', activation2.output)
print('predictions:', predictions)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print('y:', y)
print(predictions == y)
 
# Print accuracy
print('acc:', accuracy)