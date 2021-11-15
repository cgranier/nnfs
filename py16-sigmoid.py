# Plot a sigmoid curve
# Just learning how to use matplotlib

import matplotlib.pyplot as plt
import numpy as np

ax = plt.subplot()

x = np.arange(-5,5,.01)
s = 1 / (1 + np.e**(-x))
line = ax.plot(x, s)
ax.set_ylim(-0.5,1.5)
ax.grid(True)

plt.axvline(x=0, c='green')
plt.axhline(y=0, c='green')
plt.show()
