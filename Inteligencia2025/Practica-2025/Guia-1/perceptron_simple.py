import numpy as np
import matplotlib.pyplot as plt
from graficar import graficar

act_sigmoide = lambda x: (2/(1+np.exp(-x)))-1

def perceptron_simple(datos_trn, datos_tst, epocas, factor_aprendizaje, modo):
    # Separar las variables: las dos primeras columnas van a x_train, la tercera a y_train
    x_train = datos_trn[:, :2]  # Primeras dos columnas como matriz
    y_train = datos_trn[:, 2]   # Tercera columna como vector
    x_tst = datos_tst[:, :2]  # Primeras dos columnas como matriz
    y_tst = datos_tst[:, 2]   # Tercera columna como vector


    # Agregar columna de unos (bias) al inicio de x_train
    x_train = np.column_stack([-np.ones(x_train.shape[0]), x_train])
    x_tst = np.column_stack([-np.ones(x_tst.shape[0]), x_tst])

    #Entrenamiento
    w = np.random.uniform(-0.5, 0.5, x_train.shape[1])
    contador_epocas = 1
    error_tolerado = 0.10
    error_epoca = 1
    while (contador_epocas <= epocas) & (error_tolerado < error_epoca):
        error_epoca = 0
        aciertos = 0
        for i in range(len(x_train)):
            y = np.sign(act_sigmoide(w @ x_train[i]))
            if y == y_train[i]:
                aciertos += 1
            error_local = abs(y_train[i] - y)
            error_epoca += error_local
            w += factor_aprendizaje * (y_train[i] - y) * x_train[i]

        # Graficar patrones
        graficar(x_train,w, modo)

        # Calcular error promedio de la época
        error_epoca = error_epoca / len(x_train)
        print(f"Error de la época {contador_epocas}: {error_epoca}")
        print(f"Aciertos de la época {contador_epocas}: {aciertos}")
        contador_epocas += 1
    
    plt.ioff()  # Desactiva modo interactivo
    plt.show()  # Mantiene la última gráfica abierta

    #Test
    aciertos = 0
    for i in range(len(x_tst)):
        y = np.sign(act_sigmoide(w @ x_tst[i]))
        if y == y_tst[i]:
            aciertos += 1
    print(f"Aciertos del test: {aciertos}")
        
