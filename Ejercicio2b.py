import numpy as np

def gauss_seidel(A, b, x0, iter_count):
    """
    Resuelve el sistema Ax = b usando el método de Gauss-Seidel.
    
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
    n = len(b)
    x = x0.copy()
    for k in range(iter_count):
        x_new = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        x = x_new
        print(f"Iteración {k+1}: {x}")
    return x

# Datos del sistema B
A = np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]], dtype=float)
b = np.array([9, 7, 6], dtype=float)
x0 = np.zeros(3, dtype=float)

# Número de iteraciones
iter_count = 2

# Resolver el sistema
print("Sistema b:")
gauss_seidel(A, b, x0, iter_count)
