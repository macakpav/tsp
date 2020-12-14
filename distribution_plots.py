import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Distribution for plotting
n=int(5e5)

# values = np.random.normal(0.5,0.1,n)
# values = np.random.uniform(0,1,n)
values = np.random.gamma(5,1,n)

count, bins, ignored = plt.hist(values, 20, density=True)
plt.show()