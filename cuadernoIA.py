# Método de cuadrados ordinarios

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

# Cargamos la librería
boston = load_boston()

x = np.array(boston.data[:, 5])
y = np.array(boston.target)

plt.scatter(x, y, alpha=0.5)

# Añadimos columna de 1s para terminos independientes en la formula

x = np.array([np.ones(506), x]).T

B = np.linalg.inv(x.T @ x) @ x.T @ y


plt.plot([4,9], [B[0] + B[1] * 4, B[0] + B[1] * 9], c="red")

plt.show()
