import numpy as np

def gauss_seidel_tol(A, b, x0, tol):
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
    n = len(b)  # Número de ecuaciones
    x = x0.copy()  # Copia del vector inicial de estimaciones
    diff = tol + 1  # Inicializa la diferencia para el criterio de convergencia
    iter_count = 0  # Contador de iteraciones
    
    while diff > tol:  # Mientras la diferencia sea mayor que la tolerancia
        x_new = x.copy()  # Copia del vector de estimaciones para esta iteración
        for i in range(n):
            # Suma de los productos de los elementos ya actualizados
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            # Suma de los productos de los elementos no actualizados
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            # Actualización del valor de x_new[i]
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        # Calcula la diferencia máxima entre x_new y x
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new  # Actualiza el vector de estimaciones
        iter_count += 1  # Incrementa el contador de iteraciones
        print(f"Iteración {iter_count}: {x}, Error: {diff}")
    
    return x

# Datos del sistema A
A = np.array([[3, -1, 1], [3, -6, -2], [3, 3, 7]])
b = np.array([1, 0, 4])
x0 = np.zeros(3)  # Estimación inicial

# Tolerancia
tol = 1e-3

# Resolver el sistema
print("Sistema a:")
gauss_seidel_tol(A, b, x0, tol)
