import math

# An exmaple output from the outer layer of the neural network
softmax_output = [0.7, 0.1, 0.2]

# Ground truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] + 
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)

# target_output[1] and [2] are 0, so anything multiplied by them is 0
# target_output[0] is = 1 (since target_output is a one-hot target) and
# anything multiplied by 1 is itself, thus:
loss_simple = -math.log(softmax_output[0])

print(loss_simple)

print('-------')
# print(math.log(0))
print(math.log(1.))
print(math.log(0.95))
print(math.log(0.9))
print(math.log(0.8))
print('...')
print(math.log(0.2))
print(math.log(0.1))
print(math.log(0.05))
print(math.log(0.01))

# Testing the Natural Log
print(math.log(5.2))
print(2.718281829**1.64866)