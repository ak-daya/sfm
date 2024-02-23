import scipy.optimize as optimize
import numpy as np
from EstimateFundamentalMatrix import Homogenize
from LinearTriangulation import TriangulateDepth_Linear

def loss_fn(WorldPts, camera_params, X1, X2):
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

def objective_fn(x0, camera_params, X1, X2):
	WorldPts = x0
	loss = loss_fn(WorldPts, camera_params, X1, X2)
	return loss

def TriangulateDepth_NonLinear(camera_params, X1, X2, x0_worldPoints):
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
	# x0_worldPoints = TriangulateDepth_Linear(camera_params, X1, X2)
	
	# Optimize
	optimized_worldPoints = []
	print("Reprojection Error Before Optimization:")
	Loss = 0
	for i, point3d in enumerate(x0_worldPoints):
		WorldPts = point3d
		loss = loss_fn(WorldPts, camera_params, X1[i], X2[i])
		Loss += loss
	Loss = Loss/len(x0_worldPoints)
	print(Loss)
		

	print("Optimizing non-linear triangulation...")
	
	# x0_worldPoints = np.array(x0_worldPoints).reshape(-1)
	for i, point3d in enumerate(x0_worldPoints):
		x0_worldPoint = point3d
		model = optimize.least_squares(fun=objective_fn, 
						 	x0=x0_worldPoint,  
							args=[camera_params, X1[i], X2[i]]
							)
		optimized_worldPoints.append(model.x)  
	print("Optimization complete.")
	Loss = 0
	for i, point3d in enumerate(optimized_worldPoints):
		WorldPts = point3d
		loss = loss_fn(WorldPts, camera_params, X1[i], X2[i])
		Loss += loss
	Loss = Loss/len(optimized_worldPoints)
	print(Loss)#.reshape(-1, 3)

	return optimized_worldPoints