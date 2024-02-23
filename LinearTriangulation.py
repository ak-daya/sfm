import numpy as np
from EstimateFundamentalMatrix import Homogenize

def ReprojectionError(WorldPts, camera_params, X1, X2):
	K, C1, R1, C2, R2 = camera_params
	# WorldPts = WorldPts.reshape(-1, 3)

	I = np.identity(3)
	P1 = K @ R1 @ np.concatenate([I, -C1.reshape((3,1))], axis=1)
	P2 = K @ R2 @ np.concatenate([I, -C2.reshape((3,1))], axis=1)
	X = np.array([X1, X2])
	P = np.array([P1, P2])
	# num_points = WorldPts.shape[0]
	# for i in range(num_points):
	norm = 0
	for j in range(2):
		u_actual = X[j]
		u_proj = P[j] @ Homogenize(WorldPts)
		u_proj = u_proj/u_proj[-1]
		norm += np.sum((u_actual - u_proj[:-1])**2, axis=0)
	return norm


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
    Error = 0
    for i in range(num_points):
        
        X1_constraint = np.array([[0, -X1_hom[i][2], X1_hom[i][1]], [X1_hom[i][2], 0, -X1_hom[i][0]], [-X1_hom[i][1], X1_hom[i][0], 0]]) @ P1
        X2_constraint = np.array([[0, -X2_hom[i][2], X2_hom[i][1]], [X2_hom[i][2], 0, -X2_hom[i][0]], [-X2_hom[i][1], X2_hom[i][0], 0]]) @ P2

        A = np.concatenate([X1_constraint, X2_constraint], axis=0)
        _, _, Vt = np.linalg.svd(A)
        depth = Vt[-1,:]

        depth = depth/depth[-1]
        
        depth = depth[:3]

        depthPts.append(depth)
        
        Error += ReprojectionError(depth, camera_params, X1_hom[i], X2_hom[i])
    
    Error = Error/num_points

    return depthPts, Error