import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 1.0, 0.01)
values = np.random.normal(0.5,0.1,np.shape(t))
values = np.sort(values)

fig, ax = plt.subplots()
ax.plot(t, values)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()