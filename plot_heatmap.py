import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('global_grid.txt')
plt.imshow(data, cmap='hot', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of Poisson Equation Solution')
plt.savefig('heatmap.png', dpi=300)
plt.close()