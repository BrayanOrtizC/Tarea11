import numpy as np

def jacobi(A, b, x0, iter_count):
    """
    Resuelve el sistema Ax = b usando el método de Jacobi.

    Parámetros:
    A : np.ndarray
        Matriz de coeficientes.
    b : np.ndarray
        Vector de términos constantes.
    x0 : np.ndarray
        Vector de estimación inicial.
    iter_count : int
        Número de iteraciones a realizar.

    Retorna:
    np.ndarray
        Solución aproximada del sistema.
    """
    # Extrae la matriz diagonal D de A
    D = np.diag(np.diag(A))
    # Calcula la matriz R como A - D
    R = A - D
    # Inicializa la estimación con x0
    x = x0
    # Itera iter_count veces
    for i in range(iter_count):
        # Actualiza x usando la fórmula del método de Jacobi
        x = np.dot(np.linalg.inv(D), b - np.dot(R, x))
    return x

def gauss_seidel(A, b, x0, tol):
    """
    Resuelve el sistema Ax = b usando el método de Gauss-Seidel con tolerancia para la convergencia.

    Parámetros:
    A : np.ndarray
        Matriz de coeficientes.
    b : np.ndarray
        Vector de términos constantes.
    x0 : np.ndarray
        Vector de estimación inicial.
    tol : float
        Tolerancia para el criterio de convergencia.

    Retorna:
    np.ndarray
        Solución aproximada del sistema.
    """
    # Número de ecuaciones
    n = len(b)
    # Copia del vector inicial de estimaciones
    x = x0.copy()
    # Inicializa la diferencia para el criterio de convergencia
    diff = tol + 1
    # Contador de iteraciones
    iter_count = 0

    while diff > tol:  # Mientras la diferencia sea mayor que la tolerancia
        x_new = x.copy()  # Copia del vector de estimaciones para esta iteración
        for i in range(n):
            # Calcula la suma de los productos de los elementos ya actualizados
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            # Calcula la suma de los productos de los elementos no actualizados
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            # Actualiza el valor de x_new[i]
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        # Calcula la diferencia máxima entre x_new y x
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new  # Actualiza el vector de estimaciones
        iter_count += 1  # Incrementa el contador de iteraciones
    return x

# Datos del sistema
A = np.array([[2, -1, 1], [2, 2, 2], [-1, -1, 2]])
b = np.array([-1, 4, -5])
x0 = np.zeros(3)  # Estimación inicial

# Resolver el sistema usando el método de Jacobi
print("Método de Jacobi con 25 iteraciones:")
x_jacobi = jacobi(A, b, x0, 25)
print(f"Solución aproximada después de 25 iteraciones: {x_jacobi}")

# Resolver el sistema usando el método de Gauss-Seidel
print("\nMétodo de Gauss-Seidel con tolerancia 10^-5:")
tol = 1e-5
x_gauss_seidel = gauss_seidel(A, b, x0, tol)
print(f"Solución aproximada con Gauss-Seidel: {x_gauss_seidel}")
