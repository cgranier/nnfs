import matplotlib.pyplot as plt

starting_learning_rate = 1.
learning_rate_decay = 0.1

x = []
y = []

for step in range(20):
    learning_rate = starting_learning_rate * \
                    (1. / (1 + learning_rate_decay * step))
    print(learning_rate)
    y.append(step)
    x.append(learning_rate)

plt.plot(x, y)
plt.show()