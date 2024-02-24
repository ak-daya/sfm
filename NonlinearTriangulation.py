import scipy.optimize as optimize
import numpy as np
from LinAlgTools import Homogenize
from LinearTriangulation import LinearTriangulation
from Projection import *

# def loss_fn(WorldPts, camera_params, X1, X2):
# 	K, C1, R1, C2, R2 = camera_params
# 	WorldPts = WorldPts.reshape((-1,3))

# 	I = np.identity(3)
# 	P1 = K @ R1 @ np.concatenate([I, -C1], axis=1)
# 	P2 = K @ R2 @ np.concatenate([I, -C2], axis=1)
	
# 	error = []

# 	X = np.array([X1, X2])
# 	P = np.array([P1, P2])
# 	num_points = WorldPts.shape[0]
# 	for i in range(num_points):
# 		norm = 0
# 		for j in range(2):
# 			u_actual = X[j][i]
# 			u_proj = P[j] @ Homogenize(X[j][i])
# 			u_proj = u_proj/u_proj[-1]
# 			norm += np.sum((u_actual - u_proj[:-1])**2, axis=0)
# 		error.append(norm)

# 	return np.array(error)

# def objective_fn(x0, camera_params, X1, X2):
# 	WorldPts = x0
# 	loss = loss_fn(WorldPts, camera_params, X1, X2)
	
# 	return loss

# def TriangulateDepth_NonLinear(camera_params, X1, X2):
# 	"""
#     Triangulates a set of 3D points w.r.t. a camera frame at origin (0,0,0),
#     given the intrinsic matrix K, a set of correspondences X1 and X2, 
#     and a relative camera pose located at camera center C, with rotation R,
# 	using non-linear reprojection error minimization
    
#     camera_params:
#         K: Camera intrinsics matrix
#         C1: Camera center 1
#         R1: Camera rotation 1
#         C2: Camera center 2
#         R2: Camera rotation 2
#     x1: Correspondence set 1
#     x2: Correspondence set 2
#     """

# 	# Compute linear triangulation
# 	x0_worldPoints = LinearTriangulation(camera_params)
	
# 	# Optimize
# 	model = opt.least_squares(fun=objective_fn, 
# 						 	x0=x0_worldPoints, 
# 							method="lm", 
# 							args=[camera_params, X1, X2]
# 							)
# 	worldPoints = model.x.reshape((-1,3))

# 	return worldPoints

def objective_fn(x0, K, C1, R1, C2, R2, X1, X2):
	# Reshape x0
	x0 = x0.reshape((-1,3))
	P1 = GetProjectionMatrix(C1, R1, K)
	P2 = GetProjectionMatrix(C2, R2, K)
	X = np.array([X1, X2])
	P = np.array([P1, P2])

	norm = np.zeros((x0.shape[0],), dtype=np.float32)
	for j in range(2):
		u_actual = X[j]
		u_proj = World2Image(x0, P[j])
		norm += ReprojectionError(u_proj, u_actual)
	
	return norm

def NonLinearTriangulation(X0, K, C1, R1, C2, R2, X1, X2):
	"""
    Triangulates a set of 3D points w.r.t. a camera frame at origin (0,0,0),
    given the intrinsic matrix K, a set of correspondences X1 and X2, 
    and a relative camera pose located at camera center C, with rotation R,
    using linear least squares (algebraic minimization)
    
	X0: Estimated 3D World points
    K: Camera intrinsics matrix
    C1: Camera center 1 relative to itself
    R1: Camera rotation 1 relative to itself
    C: Camera center 2 relative to center 1
    R: Camera rotation 2 relative to center 1
    x1: Correspondence set 1
    x2: Correspondence set 2

    worldPts: Set of estimated 3D world points
    """
	print("Optimizing non-linear triangulation...")

	X0 = X0.ravel()
	model = optimize.least_squares(fun=objective_fn, 
						 			x0=X0, 
									method="trf",
									args=[K, C1, R1, C2, R2, X1, X2]
									)
	worldPoints = model.x.reshape((-1,3))
	
	print("Optimization complete.")

	return worldPoints