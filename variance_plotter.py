import sys
import numpy as np
import matplotlib.pyplot as plt

w = 8
h = 6
dpi = 1

variance = np.loadtxt(sys.argv[1])

fig = plt.figure(frameon=False)
fig.set_size_inches(w,h)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.matshow(-variance, cmap='binary')

fig.savefig("var.png")