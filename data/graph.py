import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('file_energy.data')

fig = plt.figure()

x = np.array([np.float64(x) for x in range(0, 116000)])
y = np.exp(0.000525 * x, dtype = np.float64)
plt.plot(y, label='exp^0.000525*x')

plt.plot(data, label = 'η = 0.1, τ = 10^-3')
plt.yscale('log')

plt.title('Magnetic field energies (log scale)')
plt.xlabel('Steps')
plt.ylabel('Energy')
plt.legend()
plt.show()