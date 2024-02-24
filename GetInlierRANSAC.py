import numpy as np
import random
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from LinAlgTools import *


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

		points1_hom = Homogenize(points1)
		points2_hom = Homogenize(points2)
		
		for j in range(num_points):
			if abs(points2_hom[j] @ F @ points1_hom[j]) < eps:
				inlier_idxs.append(j)

		if len(inlier_idxs) > max_inliers:
			# print(f"Good matches found: {len(inlier_idxs)}")
			max_inliers = len(inlier_idxs)
			best_inlier_idxs = inlier_idxs
		
		# Early break condition
		if max_inliers >= early_break_condition:
			break
	
	best_points1 = points1[best_inlier_idxs]	 
	best_points2 = points2[best_inlier_idxs]	 

	return best_points1, best_points2
