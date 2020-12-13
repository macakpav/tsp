import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_best():
    values = [27898.47885663667, 
    28147.707152356725, 
    28081.34336228344, 
    28675.361062793712, 
    27728.783400021668, 
    28675.361062793712,
    27446.882896509844,
    28153.43071394258,
    28334.617694625726
    ,27510.726640339057
    ,28345.272237299392
    ,27353.476650183235
    ,28671.06513996771
    ,29733.287243651626
    ,29724.33803886699
    ,27154.48839924464
    ,27672.47407848784
    ,27525.62576918241
    ,28019.94274765628
    ,28565.248363039416
    ,28488.007365340105]

    plt.plot(values)
    plt.title("Best values")
    plt.xticks(np.arange(0, 21, 2))
    plt.ylabel("Objective value")
    plt.savefig('best_values.png')

if __name__ == "__main__":
    colnames = ['Iteration', 'Elapsed time(s)', 'Mean value', 'Best value'] 
    frame = pd.read_csv('r0829194.csv', skiprows=2, names=colnames, usecols=[0,1,2,3])
    print(frame.head())
    frame.plot(x='Elapsed time(s)', y=['Best value', 'Mean value'])
    plt.title('Objective values')
    plt.savefig('objective_value.png')
    plt.close()
    # plot_best()


