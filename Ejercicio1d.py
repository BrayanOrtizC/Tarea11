import numpy as np

def jacobi(A, b, x0, iter_count):
    """
    Aplica el método de Jacobi para resolver el sistema Ax = b.
    
    Parámetros:
    A -- Matriz de coeficientes
    b -- Vector de constantes
    x0 -- Estimación inicial de la solución
    iter_count -- Número de iteraciones
    
    Retorna:
    x -- Solución aproximada después de iter_count iteraciones
    """
    D = np.diag(np.diag(A))  # Extraer la matriz diagonal de A
    R = A - D  # Obtener la matriz residual (A - D)
    x = x0  # Inicializar la solución con x0
    for i in range(iter_count):
        x = np.dot(np.linalg.inv(D), b - np.dot(R, x))  # Aplicar la fórmula de Jacobi
        print(f"Iteración {i+1}: {x}")  # Imprimir el estado actual de la solución
    return x

# Definir el sistema d
A_d = np.array([[4, 1, 1, 0, 1], [-1, -3, 1, 1, 0], [2, 1, 5, -1, -1], [-1, -1, -1, 4, 0], [0, 2, -1, 1, 4]])
b_d = np.array([6, 6, 6, 6, 6])
x0_d = np.zeros(5)  # Estimación inicial

# Número de iteraciones
iter_count = 2

print("\nSistema d:")
jacobi(A_d, b_d, x0_d, iter_count)  # Resolver el sistema d usando Jacobi
