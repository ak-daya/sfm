import scipy.optimize as opt
import numpy as np
from EstimateFundamentalMatrix import Homogenize
from LinearTriangulation import TriangulateDepth_Linear

def loss_fn(WorldPts, camera_params, X1, X2):
	K, C1, R1, C2, R2 = camera_params
	WorldPts = WorldPts.reshape((-1,3))

	I = np.identity(3)
	P1 = K @ R1 @ np.concatenate([I, -C1], axis=1)
	P2 = K @ R2 @ np.concatenate([I, -C2], axis=1)
	
	error = []

	X = np.array([X1, X2])
	P = np.array([P1, P2])
	num_points = WorldPts.shape[0]
	for i in range(num_points):
		norm = 0
		for j in range(2):
			u_actual = X[j][i]
			u_proj = P[j] @ Homogenize(X[j][i])
			u_proj = u_proj/u_proj[-1]
			norm += np.sum((u_actual - u_proj[:-1])**2, axis=0)
		error.append(norm)

	return np.array(error)

def objective_fn(x0, camera_params, X1, X2):
	WorldPts = x0
	loss = loss_fn(WorldPts, camera_params, X1, X2)
	
	return loss

def TriangulateDepth_NonLinear(camera_params, X1, X2):
	"""
    Triangulates a set of 3D points w.r.t. a camera frame at origin (0,0,0),
    given the intrinsic matrix K, a set of correspondences X1 and X2, 
    and a relative camera pose located at camera center C, with rotation R,
	using non-linear reprojection error minimization
    
    camera_params:
        K: Camera intrinsics matrix
        C1: Camera center 1
        R1: Camera rotation 1
        C2: Camera center 2
        R2: Camera rotation 2
    x1: Correspondence set 1
    x2: Correspondence set 2
    """

	# Compute linear triangulation
	x0_worldPoints = TriangulateDepth_Linear(camera_params)
	
	# Optimize
	model = opt.least_squares(fun=objective_fn, 
						 	x0=x0_worldPoints, 
							method="lm", 
							args=[camera_params, X1, X2]
							)
	worldPoints = model.x.reshape((-1,3))

	return worldPoints