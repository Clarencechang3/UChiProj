import numpy as np

stdevs = []
for i in range(0, 100):
    filename = "path" + str(i) + ".csv"
    returns = np.genfromtxt(filename, dtype=float, delimiter=",", usecols=(1), autostrip=True, skip_header=1)

    stdevs.append(np.std(returns, dtype=float))

print(np.mean(stdevs))