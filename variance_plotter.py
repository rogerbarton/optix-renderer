import sys
import numpy as np
import matplotlib.pyplot as plt


variance = np.loadtxt(sys.argv[1])

fig, (ax1,ax2) = plt.subplots(1, 2)

fig.suptitle("Adaptive Sampling: final probabilities")

ax1.set_title("Final Probability")
ax1.matshow(-variance, cmap='binary')

ax1.set_xticks([],[])
ax1.set_yticks([],[])

ax2.matshow(-np.log(variance / np.max(variance)), cmap='binary')
ax2.set_title("log of normalized probability")

ax2.set_xticks([],[])
ax2.set_yticks([],[])


plt.tight_layout()


plt.show()