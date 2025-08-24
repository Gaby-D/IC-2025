import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from perceptron_simple import perceptron_simple
from perceptron_simple_clase import perceptron_simple_clase

# Cargar el archivo de entrenamiento
# arch_name_trn = abspath('Inteligencia-Computacional/Practica-2025/Guia-1/data/OR_trn.csv')
# arch_name_tst = abspath('Inteligencia-Computacional/Practica-2025/Guia-1/data/OR_tst.csv')
# arch_name_trn = abspath('Inteligencia-Computacional/Practica-2025/Guia-1/data/OR_50_trn.csv')
# arch_name_tst = abspath('Inteligencia-Computacional/Practica-2025/Guia-1/data/OR_50_tst.csv')
arch_name_trn = abspath('Inteligencia-Computacional/Practica-2025/Guia-1/data/OR_90_trn.csv')
arch_name_tst = abspath('Inteligencia-Computacional/Practica-2025/Guia-1/data/OR_90_tst.csv')
# arch_name_trn = abspath('Inteligencia-Computacional/Practica-2025/Guia-1/data/XOR_trn.csv')
# arch_name_tst = abspath('Inteligencia-Computacional/Practica-2025/Guia-1/data/XOR_tst.csv')

# Leer el archivo CSV usando numpy
X_train, y_train = perceptron_simple_clase.cargar_csv(arch_name_trn)
X_test, y_test = perceptron_simple_clase.cargar_csv(arch_name_tst)

# crear perceptrón
p = perceptron_simple_clase(tasa_aprendizaje=0.0001, max_epocas = 100, tolerancia=0)

# entrenar
p.entrenar(X_train, y_train)

print("Pesos: ", p.w)
print("Precision en test: ", p.evaluar(X_test, y_test))

plt.show()