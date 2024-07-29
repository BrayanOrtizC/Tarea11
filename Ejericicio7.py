import numpy as np

def gauss_jacobi(A, b, tol=1e-15, max_iter=3000):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    
    for iteration in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, iteration + 1
        
        x = x_new.copy()
    
    raise ValueError("El método no converge después de el número máximo de iteraciones")

A = np.array([
    [1, 0, -1],
    [-1/2, 1, -1/4],
    [1, -1/2, 1]
])
b = np.array([0.2, -1.425, 2])

solucion, iteraciones = gauss_jacobi(A, b)

print(f"La solución del sistema de ecuaciones es: {solucion}")
print(f"El método convergió en {iteraciones} iteraciones")
