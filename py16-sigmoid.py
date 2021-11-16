# Plot a sigmoid curve
# Just learning how to use matplotlib

import matplotlib.pyplot as plt
import numpy as np

ax = plt.subplot()

x = np.arange(-5,5,.01)
s = 1 / (1 + np.e**(-x))
line = ax.plot(x, s)
ax.set_ylim(-4,4)
ax.grid(True, alpha=0.25)

plt.axvline(x=0, c='green', alpha=0.25)
plt.axhline(y=0, c='green', alpha=0.25)
plt.show()
