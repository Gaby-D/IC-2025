import numpy as np
import matplotlib.pyplot as plt
from graficar import graficar

# Funcion de activacion
act_sigmoide = lambda x: (2/(1+np.exp(-x)))-1

# Funcion perceptron con entrenamiento y prueba
def perceptron_simple(datos_trn, datos_tst, epocas, factor_aprendizaje, error_tolerado, modo):
    # Separar las variables: las dos primeras columnas van a x, la tercera a y
    x_train = datos_trn[:, :2]  # Primeras dos columnas como matriz
    y_train = datos_trn[:, 2]   # Tercera columna como vector
    x_tst = datos_tst[:, :2]  # Primeras dos columnas como matriz
    y_tst = datos_tst[:, 2]   # Tercera columna como vector


    # Agregar columna de unos (bias) al inicio de x_train y x_test
    x_train = np.column_stack([-np.ones(x_train.shape[0]), x_train])
    x_tst = np.column_stack([-np.ones(x_tst.shape[0]), x_tst])

    # Inicializar variables
    w = np.random.uniform(-0.5, 0.5, x_train.shape[1])
    contador_epocas = 1
    error_epoca = 100

    # Entrenamiento por corte de epocas o error encontrado
    while (contador_epocas <= epocas) & (error_tolerado < error_epoca):
        error_epoca = 0
        aciertos = 0
        for i in range(len(x_train)): # Por cada patron
            y = np.sign(act_sigmoide(w @ x_train[i])) # y(n) = sigm(<w(n); x(n)>)
            w += (0.5*factor_aprendizaje) * (y_train[i] - y) * x_train[i] # w(n + 1) = w(n) + alpha/2 * [yd(n) - y(n)] * x(n)

        # Ahora calculo errores fuera del for
        y_pred = np.sign(act_sigmoide(x_train @ w))
        errores = (y_train - y_pred) ** 2
        error_epoca = np.mean(errores)
        aciertos = np.sum(y_pred == y_train)

        # Graficar patrones y linea de separacion
        graficar(x_train, w, modo)

        print(f"Error de la época {contador_epocas}: {error_epoca}")
        print(f"Aciertos de la época {contador_epocas}: {aciertos}")
        contador_epocas += 1
    
    plt.ioff()  # Desactiva modo interactivo
    plt.show()  # Mantiene la última gráfica abierta

    # Test
    aciertos = 0
    for i in range(len(x_tst)):
        y = np.sign(act_sigmoide(w @ x_tst[i]))
        if y == y_tst[i]:
            aciertos += 1
    print(f"Aciertos del test: {aciertos} de {len(x_tst)} totales")
        
