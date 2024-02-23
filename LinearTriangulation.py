import numpy as np
from EstimateFundamentalMatrix import Homogenize
from Visualization import Plot3DPointSets

def Skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def TriangulateDepth_Linear(C, R, K, X1, X2):
    """
    Triangulates a set of 3D points w.r.t. a camera frame at origin (0,0,0),
    given the intrinsic matrix K, a set of correspondences X1 and X2, 
    and a relative camera pose located at camera center C, with rotation R,
    using linear least squares (algebraic minimization)
    
    C: Camera center 2 relative to center 1
    R: Camera rotation 2 relative to center 1
    K: Camera intrinsics matrix
    x1: Correspondence set 1
    x2: Correspondence set 2
    """
    # Homogenize
    X1_hom = Homogenize(X1)
    X2_hom = Homogenize(X2)

    # First camera frame at origin
    I = np.identity(3)
    R1 = I
    C1 = np.zeros((3,1))
    P1 = K @ R1 @ np.concatenate([I, -C1], axis=1)

    # Second camera frame
    C = C.reshape((3,1))
    P2 = K @ R @ np.concatenate([I, -C], axis=1)

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
    
    # Plot3DPointSets([depthPts], ['blue'], ['Pose'])

    return depthPts

