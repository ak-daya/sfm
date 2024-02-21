import numpy as np
from EstimateFundamentalMatrix import Homogenize

def TriangulateDepth_Linear(camera_params, X1, X2):
    """
    Triangulates a set of 3D points w.r.t. a camera frame at origin (0,0,0),
    given the intrinsic matrix K, a set of correspondences X1 and X2, 
    and a relative camera pose located at camera center C, with rotation R,
    using linear least squares (algebraic minimization)
    
    camera_params:
        K: Camera intrinsics matrix
        C1: Camera center 1
        R1: Camera rotation 1
        C2: Camera center 2
        R2: Camera rotation 2
    x1: Correspondence set 1
    x2: Correspondence set 2
    """
    K, C1, R1, C2, R2 = camera_params

    # First camera frame
    I = np.identity(3)
    P1 = K @ R1 @ np.concatenate([I, -C1.reshape((3,1))], axis=1)

    # Second camera frame
    P2 = K @ R2 @ np.concatenate([I, -C2.reshape((3,1))], axis=1)

    # Homogenize
    X1_hom = Homogenize(X1)
    X2_hom = Homogenize(X2)

    depthPts = []
    num_points = X1.shape[0]
    for i in range(num_points):
        X1_constraint = np.cross(X1_hom[i], P1)
        X2_constraint = np.cross(X2_hom[i], P2)

        A = np.concatenate([X1_constraint, X2_constraint], axis=0)
        _, _, Vt = np.linalg.svd(A)
        depth = Vt[-1,:]

        depth = depth/depth[-1]

        depthPts.append(depth)

    return depthPts