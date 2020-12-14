import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    colnames = ['Iteration', 'Elapsed time(s)', 'Mean value', 'Best value'] 
    frame = pd.read_csv('r0829194.csv', skiprows=2, names=colnames, usecols=[0,1,2,3])
    print(frame.head())
    frame.plot(x='Elapsed time(s)', y=['Best value', 'Mean value'])
    plt.title('Objective values')
    plt.savefig('objective_value.png')
    plt.close()


