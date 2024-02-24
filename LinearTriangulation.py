import numpy as np
from EstimateFundamentalMatrix import Homogenize
from Visualization import Plot3DPointSets
from Projection import *
from LinAlgTools import Skew


def LinearTriangulation(K, C1, R1, C2, R2, X1, X2):
    """
    Triangulates a set of 3D points w.r.t. a camera frame at origin (0,0,0),
    given the intrinsic matrix K, a set of correspondences X1 and X2, 
    and a relative camera pose located at camera center C, with rotation R,
    using linear least squares (algebraic minimization)
    
    K: Camera intrinsics matrix
    C1: Camera center 1 relative to itself
    R1: Camera rotation 1 relative to itself
    C: Camera center 2 relative to center 1
    R: Camera rotation 2 relative to center 1
    x1: Correspondence set 1
    x2: Correspondence set 2

    depthPts: Set of estimated 3D world points
    """
    # Homogenize
    X1_hom = Homogenize(X1)
    X2_hom = Homogenize(X2)

    # First camera frame at origin
    P1 = GetProjectionMatrix(C1, R1, K)

    # Second camera frame
    P2 = GetProjectionMatrix(C2, R2, K)

    num_points = X1.shape[0]
    depthPts = []
        
    for i in range(num_points):
        X1_constraint = (Skew(X1_hom[i]) @ P1)[:-1,:]
        X2_constraint = (Skew(X2_hom[i]) @ P2)[:-1,:]

        A = np.concatenate([X1_constraint, X2_constraint], axis=0)

        _, _, Vt = np.linalg.svd(A)
        depth = Vt[-1,:]

        depth = depth/depth[-1]

        depthPts.append(depth[:-1])

    depthPts = np.array(depthPts)

    return depthPts