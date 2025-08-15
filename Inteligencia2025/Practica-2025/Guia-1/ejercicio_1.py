import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from perceptron_simple import perceptron_simple

# Cargar el archivo de entrenamiento
""" arch_name_trn = abspath('Practica-2025/Guia-1/OR_trn.csv')
arch_name_tst = abspath('Practica-2025/Guia-1/OR_tst.csv')
arch_name_trn = abspath('Practica-2025/Guia-1/OR_50_trn.csv')
arch_name_tst = abspath('Practica-2025/Guia-1/OR_50_tst.csv')
arch_name_trn = abspath('Practica-2025/Guia-1/OR_90_trn.csv')
arch_name_tst = abspath('Practica-2025/Guia-1/OR_90_tst.csv') """
arch_name_trn = abspath('Practica-2025/Guia-1/XOR_trn.csv')
arch_name_tst = abspath('Practica-2025/Guia-1/XOR_tst.csv')

# Leer el archivo CSV usando numpy
datos_trn = np.loadtxt(arch_name_trn, delimiter=',')
datos_tst = np.loadtxt(arch_name_tst, delimiter=',')

cant_epocas = 10
factor_aprendizaje = 0.1
modo = 'xor'

perceptron_simple(datos_trn, datos_tst, cant_epocas, factor_aprendizaje, modo)