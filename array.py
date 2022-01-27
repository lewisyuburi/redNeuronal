import numpy as np
print("Array de una dimensión")
a1 = np.array([1, 2, 3])
print(a1)
print("")

print("Array de dos dimensiónes")
a2 = np.array([[1, 2, 3], [4,5,6]])
print(a2)
print("")

print("Array de tres dimensiónes")
a3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],[10, 11, 12]]])
print(a3)
print("")

print("La función 'empty' crea y devuelve una referencia a un array vacío con las dimensiones especificadas en la tupla")
dim = (3, 3)
a4 = np.empty(dim)
print(a4)
print("")

print("La función 'zeros' crea y devuelve una referencia a un array con las dimensiones especificadas en la tupla cuyos elementos son todos ceros.")
dim = (3, 4)
a5 = np.zeros(dim)
print(a5)
print("")

print("La función 'ones' crea y devuelve una referencia a un array con las dimensiones especificadas en la tupla cuyos elementos son todos unos..")
dim = (3, 4)
a6 = np.ones(dim)
print(a6)
print("")

print("La función 'full' crea y devuelve una referencia a un array con las dimensiones especificadas en la tupla cuyos elementos son todos un valor definido.")
dim = (3, 4)
a7 = np.full(dim, 9)
print(a7)
print("")

print("La función 'indentity' crea y devuelve una referencia a la matriz identidad de dimensión 'n'")
a8 = np.identity(3)
print(a8)
print("")

print("La función 'arange' crea y devuelve una referencia a un array de una dimensión cuyos elementos son la secuencia desde inicio hasta fin tomando valores cada salto.")
a9 = np.arange(10, 100, 10)
print(a9)
print("")

print("La función 'linspace' crea y devuelve una referencia a un array de una dimensión cuyos elementos son la secuencia de n valores equidistantes desde inicio hasta fin.")
a10 = np.linspace(10, 100, 10)
print(a10)
print("")

print("La función random.random' crea y devuelve una referencia a un array con las dimensiones especificadas en la tupla cuyos elementos son aleatorios.")
dim = (3, 4)
a11 = np.random.random(dim)
print(a11)
print("")
