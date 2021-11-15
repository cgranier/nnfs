import numpy as np

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

class_targets = [0, 1, 1]

for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(targ_idx, distribution)

softmax_outputs2 = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets2 = [0, 1, 1]

print(softmax_outputs2[[0, 1, 2], class_targets2])

print(softmax_outputs2[[0], 2])

print(softmax_outputs2[range(len(softmax_outputs2)), class_targets2])

print(-np.log(softmax_outputs2[range(len(softmax_outputs2)), class_targets2]))

neg_log = -np.log(softmax_outputs2[range(len(softmax_outputs2)), class_targets2])
average_loss = np.mean(neg_log)
print(average_loss)

