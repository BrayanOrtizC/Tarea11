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

# Definir el sistema a
A_a = np.array([[3, -1, 1], [3, -6, -2], [3, 3, 7]])
b_a = np.array([1, 0, 4])
x0_a = np.zeros(3)  # Estimación inicial

# Número de iteraciones
iter_count = 2

print("\nSistema a:")
jacobi(A_a, b_a, x0_a, iter_count)  # Resolver el sistema a usando Jacobi
