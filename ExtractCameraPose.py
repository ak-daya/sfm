import numpy as np

def ExtractCameraPose(E):
    """
    Args:
        E: Essential matrix
    Returns:
        Camera poses
    """
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, _, Vt = np.linalg.svd(E)
    
    # Compute the camera centers C1, C2, C3, C4
    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = C1
    C4 = C2

    # Compute the rotation matrices R1, R2, R3, R4
    R1 = U @ W @ Vt
    R2 = R1
    R3 = U @ W.T @ Vt
    R4 = R3

    # print(f"C1: {C1}")
    # print(f"C2: {C2}")
    # print(f"C3: {C3}")
    # print(f"C4: {C4}")

    # print(f"R1: {R1}")
    # print(f"R2: {R2}")
    # print(f"R3: {R3}")
    # print(f"R4: {R4}")
    
    centers = [C1, C2, C3, C4]
    rotations = [R1, R2, R3, R4]

    # Check if the determinant of each rotation matrix is negative
    for i in range(len(rotations)):
        if np.linalg.det(rotations[i]) < 0:
            rotations[i] = -rotations[i]
            centers[i] = -centers[i]

    # print("---------------------------")
    # print(f"C1: {centers[0]}")
    # print(f"C2: {centers[1]}")
    # print(f"C3: {centers[2]}")
    # print(f"C4: {centers[3]}")

    # print(f"R1: {rotations[0]}")
    # print(f"R2: {rotations[1]}")
    # print(f"R3: {rotations[2]}")
    # print(f"R4: {rotations[3]}")

    # Return the updated rotation matrices and camera centers
    return centers, rotations