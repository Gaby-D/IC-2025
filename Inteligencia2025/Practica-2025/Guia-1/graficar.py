import numpy as np
import matplotlib.pyplot as plt

plt.ion()  # Activa modo interactivo

def graficar(x, w, modo="or"):
    plt.clf()  # Limpia el gráfico anterior

    if modo.lower() == "or":
        # OR lógico: un punto es "positivo" si X o Y son positivos
        mask_rojo = (x[:, 1] > 0) | (x[:, 2] > 0)
    elif modo.lower() == "xor":
        # XOR lógico: positivo si uno es positivo y el otro negativo
        mask_rojo = ((x[:, 1] > 0) & (x[:, 2] < 0)) | ((x[:, 1] < 0) & (x[:, 2] > 0))
    else:
        raise ValueError("Modo no reconocido. Use 'or' o 'xor'.")

    # Puntos rojos
    plt.plot(x[mask_rojo, 1], x[mask_rojo, 2], 'ro')

    # Puntos azules (resto)
    plt.plot(x[~mask_rojo, 1], x[~mask_rojo, 2], 'bo')

    # Recta de separación
    X1 = np.arange(-2, 2.01, 0.01)
    X2 = w[0]/w[2] - (w[1]/w[2]) * X1
    plt.plot(X1, X2, 'k-')

    plt.title(f"Patrones y línea de separación ({modo.upper()})")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
    plt.pause(0.01)  # Pausa breve para actualizar
