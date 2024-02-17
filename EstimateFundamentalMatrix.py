# Fundamental matrix is a 3x3 matrix (rank = 2) describing the epipolar geometry of stereo view
# To find F we need at least 8 pairs of points x1, x2
# x2.T @ F @ x1 = 0
# Find it by solving an algebraic least squares problem Ax = 0

import numpy as np
import random

def NormalizeCoordinates(homogenizedPoints):
	centroid = np.mean(homogenizedPoints, axis=0)
	translation = np.array([[1, 0, centroid[0]], [0, 1, centroid[1]], [0, 0, 1]])
	scale_u, scale_v, _ = np.std(homogenizedPoints - centroid, axis=0)
	transform = np.array([[scale_u, 0, 0],[0, scale_v, 0],[0, 0, 1]]) @ translation
	normalizedPoints = homogenizedPoints @ transform.T
	
	return normalizedPoints, transform

def Homogenize(coordinates):
	# Adds a dimension of 1
	if len(coordinates.shape) == 1:
		hom_coordinates = np.ones((coordinates.shape[0] + 1))
		hom_coordinates[:-1] = coordinates
	else:
		hom_coordinates = np.ones((coordinates.shape[0], coordinates.shape[1]+1))
		hom_coordinates[:, :-1] = coordinates
	return hom_coordinates

def EstimateFundamentalMatrix(points1, points2):
	"""
	Arguments:
		Points1: Homogeneous corr. points (N, 3)
		Points2: Homogeneous corr. points (N, 3)
	Returns:
		F: Fundamental matrix (3,3)
	"""
	assert len(points1) == len(points2), "Number of correspondences are unequal, using least common set"
	n = np.min([len(points1), len(points2)])
	points1 = points1[:n]
	points2 = points2[:n]
	
	# If normalized
	# points1, T1 = NormalizeCoordinates(points1)
	# points2, T2 = NormalizeCoordinates(points2)

	# Construct A matrix
	A = []
	for i in range(n):
			x1, y1, _ = points1[i]
			x2, y2, _ = points2[i]
			A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
	A = np.array(A)

	# SVD and enforce F rank = 2 by setting A's last singular value = 0
	_, _, V_T = np.linalg.svd(A)
	F = V_T[:, -1].reshape((3,3))

	U, S, V_T = np.linalg.svd(F, full_matrices=False)
	S[-1] = 0
	F = U @ np.diag(S) @ V_T

	# If normalized
	# F = T2.T @ F @ T1
	
	return F

def OutlierRejectionRANSAC(points1, points2, iter=1000, eps=0.05, break_percentage=0.9):
	max_inliers = 0
	best_inlier_idxs = []
	num_points = points1.shape[0]
	early_break_condition = round(break_percentage * num_points)
	
	for _ in range(iter):
		# Choose 8 correspondences randomly
		rand_idx = random.sample(range(num_points), 8)
		points1_sample = points1[rand_idx]
		points2_sample = points2[rand_idx]

		F = EstimateFundamentalMatrix(points1_sample, points2_sample)
		inlier_idxs = []
		
		for j in range(num_points):
			if abs(points2[j] @ F @ points1[j]) < eps:
				inlier_idxs.append(j)

		if len(inlier_idxs) > max_inliers:
			max_inliers = len(inlier_idxs)
			best_inlier_idxs = inlier_idxs
		
		# Early break condition
		if max_inliers >= early_break_condition:
			break
	
	best_points1 = points1[best_inlier_idxs]	 
	best_points2 = points2[best_inlier_idxs]	 

	return best_points1, best_points2

def main():
		
		x = Homogenize(np.rint(np.random.rand(10,2)*10))
		xPrime = Homogenize(np.rint(np.random.rand(10,2)*10))
		
		# x_pruned, xPrime_pruned = OutlierRejectionRANSAC(x, xPrime)

		F = EstimateFundamentalMatrix(x, xPrime)
		# F_new = EstimateFundamentalMatrix(x_pruned, xPrime_pruned)

		print(F)
		print("----")
		# print(F_new)

		# F = EstimateFundamentalMatrix(x, xPrime)
		# print(np.linalg.matrix_rank(F))
		# # Validation
		# print("Validation")
		# test_idx = random.sample(range(0, len(x)), len(x)//2)
		# for i in test_idx:
		# 		x1 = (x[i, :])
		# 		x2 = (xPrime[i, :])
		# 		res = x2 @ F @ x1
		# 		print(f"Result: {res}")

if __name__ == "__main__":
		main()