import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    n, m = len(A), len(A[0])
    A_t = np.empty((m,n), dtype = type(A[0][0]))
    for i in range(n):
        for j in range(m):
            A_t[j][i] = A[i][j]
    return A_t
