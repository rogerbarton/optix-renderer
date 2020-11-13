import sys
import numpy as np
import matplotlib.pyplot as plt


variance = np.loadtxt(sys.argv[1])

plt.matshow(variance)

plt.colorbar()


plt.show()