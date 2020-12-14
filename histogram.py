import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    colnames = ['Run', 'Iterations', 'Time', 'Mean value', 'Best value']
    if sys.argc == 1:
        csv_name = "r0829194.csv"
    else:
        csv_name = sys.argv[1]
    frame = pd.read_csv(csv_name, skiprows=1,
                        names=colnames, usecols=[0, 1, 2, 3, 4])
    # print(frame.head())
    no_runs = int(frame.tail(1)['Run'])
    # print(no_runs)
    count, bins, ignored = plt.hist(frame['Best value'], 20, density=False)
    plt.yticks(np.arange(0, max(count)+1, 5))
    plt.title("Histogram of results, best: " + str(int(min(frame['Best value']))) )
    plt.savefig('histogram.png')
    # plt.show()
    plt.close()
