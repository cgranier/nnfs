import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

print(len(class_targets.shape))
print(class_targets.shape)
print(len([[1, 0, 0]]))

# Probabilities for target_values
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
# Mask values only for one-hot encoded labels
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs*class_targets, axis=1)

# Losses
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)

print(average_loss)

print(-np.log(1e-7))
print(1e-1)
print(10**-7)