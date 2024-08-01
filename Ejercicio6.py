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
    for i in range(len(A)):
        suma = sum(abs(A[i][j]) for j in range(len(A)) if i != j)
        if abs(A[i][i]) <= suma:
            return False
    return True

def gauss_seidel(A, b, x0, tol, max_iter):
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
    max_iter : int
        Número máximo de iteraciones permitidas.

    Retorna:
    np.ndarray
        Solución aproximada del sistema.
    """
    n = len(b)
    x = x0.copy()
    diff = tol + 1
    iter_count = 0
    # Almacena la última solución calculada
    last_x = x.copy()

    while diff > tol and iter_count < max_iter:
        x_new = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        iter_count += 1
        # Actualiza la última solución calculada
        last_x = x.copy()

    # Solo imprime la última solución
    print(f"Iteración final {iter_count}: {last_x}, Error final: {diff}")
    return x

# Datos del primer sistema
A1 = np.array([[1, 0, -1], [-1/2, 1, -1/4], [1, -1/2, 1]])
b1 = np.array([0.2, -1.425, 2])
x0 = np.zeros(3)  # Estimación inicial
tol = 1e-22  # Tolerancia para la convergencia
max_iter = 300  # Número máximo de iteraciones

# Datos del segundo sistema
A2 = np.array([[1, 0, -2], [-1/2, 1, -1/4], [1, -1/2, 1]])
b2 = np.array([0.2, -1.425, 2])

# Verifica si A1 es diagonalmente dominante
print("Verificando diagonal estrictamente dominante de A1:")
dominante_A1 = es_diagonal_dominante(A1)
print(f"A1 es diagonal estrictamente dominante: {dominante_A1}")

# Resuelve el sistema A1 usando el método de Gauss-Seidel
print("\nMétodo de Gauss-Seidel para el sistema original con alta tolerancia:")
x_gauss_seidel_A1 = gauss_seidel(A1, b1, x0, tol, max_iter)
print(f"Solución aproximada para el sistema original: {x_gauss_seidel_A1}")

# Resuelve el sistema A2 usando el método de Gauss-Seidel
print("\nMétodo de Gauss-Seidel para el sistema modificado:")
x_gauss_seidel_A2 = gauss_seidel(A2, b2, x0, tol, max_iter)
print(f"Solución aproximada para el sistema modificado: {x_gauss_seidel_A2}")
