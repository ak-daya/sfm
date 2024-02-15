import numpy as np

def EssentialMatrix(K, F):
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    # Correcting the singular values of E
    S_corrected = np.array([1, 1, 0])  # Setting the third singular value to 0
    E_corrected = U @ np.diag(S_corrected) @ Vt
    return E_corrected

