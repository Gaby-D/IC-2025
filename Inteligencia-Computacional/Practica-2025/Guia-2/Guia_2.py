import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from perceptron_multicapa import perceptron_multicapa
from funciones_de_activacion import sgn
from utils import funcionLinealPerceptron, winner_take_all
from graficar import graficar

# def guia2_ejer1():
# arch_name_trn = abspath('D:\Cosas de la facu\Inteligencia Computacional\Inteligencia-2025\IC-2025\Inteligencia-Computacional\Practica-2025\Guia-1\data\XOR_trn.csv')
# arch_name_tst = abspath('D:\Cosas de la facu\Inteligencia Computacional\Inteligencia-2025\IC-2025\Inteligencia-Computacional\Practica-2025\Guia-1\data\XOR_tst.csv')
# --- Datos ejer 2 ---
arch_name_trn = abspath('D:\Cosas de la facu\Inteligencia Computacional\Inteligencia-2025\IC-2025\Inteligencia-Computacional\Practica-2025\Guia-2\data\concent_trn.csv')
arch_name_tst = abspath('D:\Cosas de la facu\Inteligencia Computacional\Inteligencia-2025\IC-2025\Inteligencia-Computacional\Practica-2025\Guia-2\data\concent_tst.csv')

num_max_epox = 1500
tolerancia = 0.30

# --- Entrenamiento ---------------------------------------------------------------------
data = np.genfromtxt(arch_name_trn, delimiter= ',')
x = data[:,0:2]
x_trn = data[:,0:2]
d = data[:,-1]
_, num_inputs = x.shape

# mlp = perceptron_multicapa(num_inputs, [2,1], 0.005)
mlp = perceptron_multicapa(num_inputs, [8,2], 0.2)

epoc = 0
error = mlp.error(x,d)
vec_error = []
while (error>tolerancia and epoc<num_max_epox):
    print(f"{epoc}: {error}")
    mlp.trn(x,d)
    error = mlp.error(x,d,"error_cuadratico")
    vec_error.append(error)
    epoc+=1

# --- test ------------------------------------------------------------------------------
data = np.genfromtxt(arch_name_tst, delimiter=',')
x = data[:, 0:2]
d = data[:, -1]

ws = mlp.obtener_pesos()

# -- Descomentar para observar las dos lineas que dividen el espacio en una arq: [2,1] --
# graficar(x_trn, modo="xor")
# funcionLinealPerceptron([ws[0][0]])
# funcionLinealPerceptron([ws[0][1]])

# -- Descomentar para caso del Percentile
for i in range(len(x)):
    y_pred = mlp.eval(x[i, :])        # salida de la red
    y_pred_scalar = y_pred[0]         # como es una sola neurona, tomo el primer valor

    if sgn(y_pred_scalar) != d[i]:
        plt.plot(x[i,0], x[i,1], '*b')   # mal clasificado
    elif y_pred_scalar > 0:
        plt.plot(x[i,0], x[i,1], '*k')   # clase positiva
    else:
        plt.plot(x[i,0], x[i,1], '*r')   # clase negativa

plt.title(f"Resultados después de la época {epoc}")

plt.figure()
plt.plot(vec_error, marker="o")
# Eje x solo numeros enteros
plt.xticks(range(0, len(vec_error), 5))

plt.title("Avance del error por época")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.grid(True)
plt.show()

print('El error cuadratico medio es:', mlp.error(x,d))