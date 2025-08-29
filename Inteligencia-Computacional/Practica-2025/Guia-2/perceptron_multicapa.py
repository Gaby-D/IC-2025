
import numpy as np
from funciones_de_activacion import sigmoide, sgn_vec
from utils import winner_take_all

class perceptron_multicapa(): 
    def __init__ (self,
                dim_patron:int,
                arquitectura:list,
                tasa_aprendizaje:float,
                fun=sigmoide
                ): 
        """
        Parametros: 
        - dim_patron: cantidad de elementos de cada patron.
        - arquitectura: lista con la cantidad de neuronas en cada capa.
        - tasa_aprendizaje (η): vel. con la que nuestra neurona aprendera.
        """
        self.__arquitectura = arquitectura
        self.__tasa_aprendizaje = tasa_aprendizaje
        self.__capas = len(arquitectura)
        self.__fun = fun
        self.__pesos = []

        for i in range(len(arquitectura)):
            # se agrega el bias y se generan los pesos aleatorios
            w = np.random.rand(self.__arquitectura[i], dim_patron+1)-0.5
            self.__pesos.append(w)
            # Para el siguiente paso, dim_patron tendra el tamaño de la salida de la capa actual
            dim_patron = arquitectura[i]

    # agrega el bias a una entrada
    def __agregar_bias(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(([-1], x))

    # calculo la propagacion obteniendo una lista con todas las salidas activadas
    def __propagacion(self, x_in: np.ndarray) -> list:
        x = x_in.copy()
        ys = []
        for i in range(self.__capas):
            x = self.__agregar_bias(x)
            x = self.__fun(self.__pesos[i] @ x)
            ys.append(x.copy())
        return ys
    
    # para obtener la salida final del multicapa
    def eval(self, x_in: np.ndarray):
        return self.__propagacion(x_in)[-1]

    # retopropagacion que calcula el gradiente
    def __retopropagacion(self, y: np.ndarray, d: np.ndarray) -> list:
        dw = []
        # capa de salida
        di = 0.5 * (d-y[-1]) * (1-y[-1]) * (1+y[-1])
        dw.append(di.copy())
        # capas ocultas
        for i in range(self.__capas-1, 0, -1):
            di = 0.5 * (np.transpose(self.__pesos[i][:, 1::]) @ di) * (1-y[i-1]) * (1+y[i-1])
            dw.insert(0, di.copy())
        return dw

    # recalcula los pesos
    def __ajuste_pesos(self, ys: np.ndarray, delta: list):
        for i in range(self.__capas):
            y = self.__agregar_bias(ys[i])
            dw = self.__tasa_aprendizaje * (np.transpose([delta[i]]) @ [y])
            self.__pesos[i] += dw
    
    # funcion de entrenamiento por lotes
    def trn(self, x_in, d):
        for i in range(len(x_in)):
            ys = self.__propagacion(x_in[i])
            dw = self.__retopropagacion(ys, d[i])
            ys.insert(0, x_in[i].copy())
            self.__ajuste_pesos(ys, dw)

    # funcion para calcular el porcentaje de error cuadratico total
    def error(self, data_set: np.ndarray, y_d: np.ndarray, etype: str = "error_cuadratico") -> float:
        err_tot = 0
        for i in range(len(data_set)):
            if etype == "error_cuadratico":
                err_tot += sum((y_d[i] - self.eval(data_set[i]))**2)
            elif etype == "porcentaje_error":
                # if np.all(np.equal(y_d[i], winner_take_all(self.eval(data_set[i])))): err_tot+=1
                if np.all(np.not_equal(y_d[i], sgn_vec(self.eval(data_set[i])))): err_tot+=1
            else:
                raise ValueError(f"{etype} no es un tipo de error valido")
            
        err = err_tot/len(data_set)
        return err 

    def obtener_pesos(self) -> list:
        return self.__pesos

    def __str__(self) -> str:
        return f"la arquitectura del modelo es: {self.__arquitectura}\ncon una profundidad de {len(self.__arquitectura)}"