# Anotaciones de la Guia 1
## Ejercicio 1
`ejercicios_1_2_3.py` contiene la lectura desde la base de datos como asi tambien el seteo de opciones para llamar a `perceptron_simple.py`, cuyo codigo entrena y realiza el test correspondiente a la neurona.

## Ejercicio 2
`graficar.py` se ejecuta en base a los valores de los patrones como asi tambien los pesos correspondiente a cada dimension del mismo.

La recta de separacion se ajusta a la ecuacion *X2 = w[0]/w[2] - (w[1]/w[2]) * X1*.

Patrones con bajas desviaciones permiten entrenar una neurona para el caso del "OR" con error nulo.

Para el caso "XOR" y bajas desviaciones en las muestras, se puede llegar a un minimo del 25% del error donde 1 de cada 4 casos sera mal clasificado.

## Ejercicio 3
Para el caso de datos generados con una desviacion aleatoria del 50%, aun permiten entrenar una neurona con error nulo.

Para el caso de datos generados con una desviacion aleatoria del 90%, se llega a un minimo de 22-25% de error cuadratico. Como error de clasificacion esto se traduce en un 5%

## A tener en cuenta a la hora de calcular error y aciertos:
- Porcentaje de error de clasificación (el que da por ejemplo 944/1000 ≈ 5.6%):

Comparando y == y_train[i] → se obtiene los aciertos.

Haciendo aciertos / len(x_train) → se obtiene el accuracy.

- Error cuadrático medio (MSE) (el que se imprime como error_epoca):

Al usar y = np.sign(...), la salida es ±1, al igual que y_train también son ±1.

Entonces:

Si y == y_train[i] → (y_train[i] - y)^2 = 0

Si y != y_train[i] → diferencia = ±2 → cuadrado = 4

Por eso, si tenés un 5.6% de errores de clasificación, el MSE ≈ 0.056 × 4 = 0.224

- Resumen:

Accuracy → 94.4%

Error de clasificación → 5.6%

MSE (con etiquetas ±1) → ~0.22