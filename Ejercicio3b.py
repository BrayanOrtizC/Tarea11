import numpy as np

def jacobi_tol(A, b, x0, tol):
    """
    Resuelve el sistema Ax = b usando el método de Jacobi con tolerancia para la convergencia.
    
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
    D = np.diag(np.diag(A))
    R = A - D
    x = x0
    diff = tol + 1
    iter_count = 0
    while diff > tol:
        x_new = np.dot(np.linalg.inv(D), b - np.dot(R, x))
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        iter_count += 1
        print(f"Iteración {iter_count}: {x}, Error: {diff}")
    return x

# Datos del sistema B
A = np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]])
b = np.array([9, 7, 6])
x0 = np.zeros(3)

# Tolerancia
tol = 1e-3

# Resolver el sistema
print("Sistema b:")
jacobi_tol(A, b, x0, tol)
