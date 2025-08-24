import numpy as np
import matplotlib.pyplot as plt
from graficar import graficar

class perceptron_simple_clase: 
  def __init__ (self, 
                tasa_aprendizaje=0.1, 
                max_epocas = 100, 
                tolerancia=0): 
    """
    Parametros: 
    - tasa_aprendizaje (η): vel. con la que nuestra neurona aprendera.
    - max_epocas: Epocas maximas de entrenamiento.
    - tolerancia: criterio de parada anticipada.
    """
    self.tasa = tasa_aprendizaje
    self.max_epocas = max_epocas
    self.tolerancia = tolerancia
   
    # Pesos del modelo    
    self.w = None # pesos
    self.errores = []
    
    # Guardamos el historial de pesos
    self.hist_w = []

  # El _ es porque es una funcion interna del metodo
  def _funcion_activacion(self, z): 
    # Funcion signo(z)
    # return np.where(z >= 0, 1, -1)  # sgn(z)
    # Funcion sigmoidea(z)
    return (2 / (1 + np.exp(-z))) - 1

  def entrenar(self, X, y): 
    
    # Añadimos el bias como primer columna fija x0 = -1
    # Con el hstack concatenamos el bias en cada uno de los patrones
    X = np.hstack([ -np.ones((X.shape[0], 1)), X ]) 

    # Inicializamos los pesos aleatorios entre [-0.5, 0.5]
    self.w = np.random.uniform(-0.5, 0.5, X.shape[1])
    self.hist_w.append(self.w) 


    # Entrenamiento por epocas
    for epoca in range(self.max_epocas): 
      aciertos = 0
      errores = 0
      for xi, objetivo in zip(X, y): 
        
        # Salida lineal -> np.dot(self.w, xi)
        # Salida no lineal -> self._funcion_activacion(np.dot(self.w, xi))
        salida = np.sign(self._funcion_activacion(np.dot(self.w, xi)))

        # Actualizamos los pesos
        self.w += self.tasa * (objetivo - salida) * xi
        self.hist_w.append(self.w.copy()) 

      
      # Ahora calculo errores fuera del for
      y_pred = np.sign(self._funcion_activacion(X @ self.w))
      
      # Error cuadratico (entre 0 y 4)
      # errores = (y - y_pred) ** 2
      # errores = np.mean(errores)

      # Error relativo (entre 0 y 1)
      errores = np.sum(y_pred != y)
      errores = errores / len(y)

      # Aciertos 
      aciertos = np.sum(y_pred == y)

      plt.ion()
      self.graficar(X, self.w, "or")
      plt.ioff()

      print(f"Época {epoca+1:3d} | Errores: {errores:6.3f} | Aciertos: {aciertos:3d} de {X.shape[0]}")
      self.errores.append(errores)
      # Criterio de finalizacion
      if errores <= self.tolerancia: 
        break
      
  def predecir(self, X): 
    
    # Añadimos la columna de bias x0 = -1 para que coincida con la dimensión de w
     X = np.hstack([ -np.ones((X.shape[0], 1)), X ])
     output = np.dot(X, self.w)
     return np.sign(self._funcion_activacion(output))
  
  def evaluar(self, X, y): 
    pred = self.predecir(X)

    # nos devuelve el porcentaje de aciertos 
    return np.mean(pred == y)

  def graficar(self, x, w, modo="or"):
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
    plt.plot(x[mask_rojo, 1], x[mask_rojo, 2], 'ro', label = "Negativos")

    # Puntos azules (resto)
    plt.plot(x[~mask_rojo, 1], x[~mask_rojo, 2], 'bo', label = "Positivos")

    # Recta de separación
    X1 = np.arange(-2, 2.01, 0.01)
    X2 = w[0]/w[2] - (w[1]/w[2]) * X1 
    plt.plot(X1, X2, 'k-')

    # Cofiguraciones del grafico
    plt.axhline(0, color='gray', linewidth=1)
    plt.axvline(0, color='gray', linewidth=1)
    plt.title(f"Patrones y línea de separación ({modo.upper()})")
    plt.xlabel("X_p")
    plt.ylabel("Y_p")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.pause(0.01)  # Pausa breve para actualizar

  
  @staticmethod
  def cargar_csv(ruta): 
    datos = np.loadtxt(ruta, delimiter=',')
    # Obtenemos todas las columnas menos la ultima
    # con el .values lo convertimos en un array de Numpy
    X = datos[:, :2]
    # Solo obtenemos la ultima
    y = datos[:, 2]

    return X, y