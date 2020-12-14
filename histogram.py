import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    colnames = ['Run', 'Iterations', 'Time', 'Mean value', 'Best value']
    if len(sys.argv) == 1:
        csv_name = "histogram_example.csv"
    else:
        csv_name = sys.argv[1]
    frame = pd.read_csv(csv_name, skiprows=2,
                        names=colnames, usecols=[0, 1, 2, 3, 4])
    # print(frame.head())
    no_runs = int(frame.tail(1)['Run'])
    best_val = min(frame['Best value'])
    avg_runtime = sum(frame['Time'])/no_runs
    # print(no_runs)
    count, bins, ignored = plt.hist(frame['Best value'], 15, density=False)
    binplt = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    plt.plot(binplt, count, linewidth=2, color='r') 
    plt.title("Histogram of results, Average runtime: " + str(int(avg_runtime)) + " s" )
    plt.xticks(bins,rotation=45)
    plt.ylabel("Number of runs in range")
    plt.xlabel("Final best value")
    plt.savefig('histogram.png')
    plt.show()
    plt.close()
