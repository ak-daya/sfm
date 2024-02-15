import numpy as np

def CameraPoseEstimation(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, S, Vt = np.linalg.svd(E)
    # Compute the camera centers C1, C2, C3, C4
    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = U[:, 2]
    C4 = -U[:, 2]

    # Compute the rotation matrices R1, R2, R3, R4
    R1 = U @ W @ Vt
    R2 = U @ W @ Vt
    R3 = U @ W.T @ Vt
    R4 = U @ W.T @ Vt
    # Check if the determinant of each rotation matrix is negative
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C2 = -C2
    if np.linalg.det(R3) < 0:
        R3 = -R3
        C3 = -C3
    if np.linalg.det(R4) < 0:
        R4 = -R4
        C4 = -C4
    # Return the updated rotation matrices and camera centers
    return [R1, R2, R3, R4], [C1, C2, C3, C4]