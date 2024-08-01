import numpy as np

def es_diagonal_dominante(A):
    """
    Verifica si una matriz es diagonalmente dominante.

    Parámetros:
    A : np.ndarray
        Matriz a verificar.

    Retorna:
    bool
        Verdadero si la matriz es diagonalmente dominante, falso en caso contrario.
    """
    # Recorre cada fila de la matriz
    for i in range(len(A)):
        # Calcula la suma de los valores absolutos de los elementos no diagonales
        suma = sum(abs(A[i][j]) for j in range(len(A)) if i != j)
        # Verifica si el valor absoluto del elemento diagonal es menor o igual a la suma calculada
        if abs(A[i][i]) <= suma:
            return False
    return True

def jacobi(A, b, x0, tol, max_iter=1000):
    """
    Resuelve el sistema Ax = b usando el método de Jacobi con tolerancia y límite de iteraciones.

    Parámetros:
    A : np.ndarray
        Matriz de coeficientes.
    b : np.ndarray
        Vector de términos constantes.
    x0 : np.ndarray
        Vector de estimación inicial.
    tol : float
        Tolerancia para el criterio de convergencia.
    max_iter : int, opcional
        Número máximo de iteraciones permitidas (por defecto es 1000).

    Retorna:
    np.ndarray
        Solución aproximada del sistema.
    """
    # Extrae la matriz diagonal D y la matriz residual R
    D = np.diag(np.diag(A))
    R = A - D
    # Inicializa la solución con el valor inicial x0
    x = x0
    iter_count = 0
    # Bucle principal del método de Jacobi
    while True:
        # Calcula la nueva solución x usando la fórmula de Jacobi
        x_new = np.dot(np.linalg.inv(D), b - np.dot(R, x))
        # Calcula la diferencia entre la solución actual y la nueva solución
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        # Actualiza la solución
        x = x_new
        iter_count += 1
        # Sale del bucle si la diferencia es menor que la tolerancia o si se alcanzó el número máximo de iteraciones
        if diff < tol or iter_count >= max_iter:
            break
    return x

def gauss_seidel(A, b, x0, tol, max_iter=1000):
    """
    Resuelve el sistema Ax = b usando el método de Gauss-Seidel con tolerancia y límite de iteraciones.

    Parámetros:
    A : np.ndarray
        Matriz de coeficientes.
    b : np.ndarray
        Vector de términos constantes.
    x0 : np.ndarray
        Vector de estimación inicial.
    tol : float
        Tolerancia para el criterio de convergencia.
    max_iter : int, opcional
        Número máximo de iteraciones permitidas (por defecto es 1000).

    Retorna:
    np.ndarray
        Solución aproximada del sistema.
    """
    # Número de variables en el sistema
    n = len(b)
    # Inicializa la solución con el valor inicial x0
    x = x0.copy()
    iter_count = 0
    # Bucle principal del método de Gauss-Seidel
    while True:
        # Copia de la solución actual para actualizar los valores en la iteración
        x_new = x.copy()
        # Recorre cada fila de la matriz
        for i in range(n):
            # Suma los productos de los elementos ya calculados de la solución
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            # Suma los productos de los elementos aún no calculados
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            # Actualiza el valor de x_new[i] usando la fórmula de Gauss-Seidel
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        # Calcula la diferencia entre la solución actual y la nueva solución
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        # Actualiza la solución
        x = x_new
        iter_count += 1
        # Sale del bucle si la diferencia es menor que la tolerancia o si se alcanzó el número máximo de iteraciones
        if diff < tol or iter_count >= max_iter:
            break
    return x

# Matriz de coeficientes para el sistema de ecuaciones
A = np.array([
    [4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 4, 0, -1, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 4, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 4, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 4, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 4, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, -1, 0, 4, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 4, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 4, -1],
    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 4],
])

# Vector de términos constantes
b = np.array([110, 110, 220, 110, 110, 110, 110, 220, 110, 110, 220, 110])
# Estimación inicial de la solución
x0 = np.zeros(len(b))
# Tolerancia para la convergencia
tol = 1e-2

# Verifica si la matriz A es diagonalmente dominante
print("Verificando diagonal estrictamente dominante de A:")
dominante_A = es_diagonal_dominante(A)
print(f"A es diagonal estrictamente dominante: {dominante_A}")

# Resuelve el sistema usando el método de Jacobi
print("\nMétodo de Jacobi con TOL = 10^-2:")
x_jacobi = jacobi(A, b, x0, tol)
print(f"Solución aproximada por Jacobi: {x_jacobi}")

# Resuelve el sistema usando el método de Gauss-Seidel
print("\nMétodo de Gauss-Seidel con TOL = 10^-2:")
x_gauss_seidel = gauss_seidel(A, b, x0, tol)
print(f"Solución aproximada por Gauss-Seidel: {x_gauss_seidel}")
