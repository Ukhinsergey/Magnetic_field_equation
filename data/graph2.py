import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('file_energy2.data')

fig = plt.figure()

x = np.array([np.float64(x) for x in range(0, 116000)])
x = x / 10000
y = np.exp(-1.75 * x, dtype = np.float64)
plt.plot(y, label='exp^-1.75*x')

plt.plot(data, label = 'η = 1, τ = 10^-4')
plt.yscale('log')

plt.title('Magnetic field energies (log scale)')
plt.xlabel('Time(sec)')
plt.ylabel('Energy')
plt.legend()
locs, labels = plt.xticks()
newlocs = locs / 10000
plt.xticks(locs[1:-1], newlocs[1:])
plt.show()