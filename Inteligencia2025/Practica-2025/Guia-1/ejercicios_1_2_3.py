import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from perceptron_simple import perceptron_simple

# Cargar el archivo de entrenamiento
arch_name_trn = abspath('Inteligencia2025/Practica-2025/Guia-1/OR_trn.csv')
arch_name_tst = abspath('Inteligencia2025/Practica-2025/Guia-1/OR_tst.csv')
# arch_name_trn = abspath('Inteligencia2025/Practica-2025/Guia-1/OR_50_trn.csv')
# arch_name_tst = abspath('Inteligencia2025/Practica-2025/Guia-1/OR_50_tst.csv')
# arch_name_trn = abspath('Inteligencia2025/Practica-2025/Guia-1/OR_90_trn.csv')
# arch_name_tst = abspath('Inteligencia2025/Practica-2025/Guia-1/OR_90_tst.csv')
# arch_name_trn = abspath('Inteligencia2025/Practica-2025/Guia-1/XOR_trn.csv')
# arch_name_tst = abspath('Inteligencia2025/Practica-2025/Guia-1/XOR_tst.csv')

# Leer el archivo CSV usando numpy
datos_trn = np.loadtxt(arch_name_trn, delimiter=',')
datos_tst = np.loadtxt(arch_name_tst, delimiter=',')

# Seleccion de propiedades
cant_epocas = 100
factor_aprendizaje = 0.01
error_tolerado = 0.10 # Al usar error cuadratico tener en cuenta que error_tolerado = error_buscado*4
modo = 'or'

perceptron_simple(datos_trn, datos_tst, cant_epocas, factor_aprendizaje, error_tolerado, modo)