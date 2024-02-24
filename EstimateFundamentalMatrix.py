import numpy as np
import cv2
import matplotlib.pyplot as plt
from LinAlgTools import *

def NormalizeCoordinates(points):
	centroid = np.mean(points, axis=0)
	scale = 2*points.shape[0]/(((points-centroid)**2).sum(axis=1).sum(axis=0))
	translation = np.array([[1, 0, -centroid[0]], [0, 1, -centroid[1]], [0, 0, 1]])
	transform = np.array([[scale, 0, 0],[0, scale, 0],[0, 0, 1]]) @ translation
	normalizedPoints = Homogenize(points) @ transform.T
	
	return normalizedPoints, transform

def EstimateFundamentalMatrix(points1, points2, normalize=False):
	"""
	Arguments:
		Points1: Homogeneous corr. points (N, 3)
		Points2: Homogeneous corr. points (N, 3)
	Returns:
		F: Fundamental matrix (3,3) s.t. epipolar constraint is met
	"""
	n = points1.shape[0]
	
	# Preconditioning: normalization and homogenization
	if normalize:
		points1, T1 = NormalizeCoordinates(points1)
		points2, T2 = NormalizeCoordinates(points2)
	else:
		points1 = Homogenize(points1)
		points2 = Homogenize(points2)

	# Construct A matrix
	A = []
	for i in range(n):
			x1, y1, _ = points1[i]
			x2, y2, _ = points2[i]
			A.append([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])
			# [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1
	A = np.array(A)

	# Find F_hat by least squares of A 
	_, _, V_T = np.linalg.svd(A)
	F_hat = V_T[-1, :].reshape((3,3))

	# Applying rank constraint
	U, S, V_T = np.linalg.svd(F_hat)
	S[-1] = 0
	F_hat = U @ np.diag(S) @ V_T

	# De-normalize F_hat
	if normalize:
		F = T2.T @ F_hat @ T1
	else:
		F = F_hat
	
	return F

def GetEpipolarPoints(F):
	"""
	F: 3x3 fundamental matrix
	Return:
	
	"""
	U, _, V_T = np.linalg.svd(F)

	epipole1 = V_T[-1,:]
	epipole2 = U[:,-1]

	return epipole1, epipole2

def GetEpilines(F, points1, points2):
	"""
	pointx: point correspondences (2,)
	F: 3 x 3 fundamental matrix

	"""
	epiline1 = points1 @ F.T
	epiline2 = points2 @ F
	return epiline1, epiline2

# Debugging not used:
def Drawlines(img1,img2,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
	_,c = img1.shape
	img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
		img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
		img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
	return img1,img2

def FindEpilines(F, img1, img2, pts1, pts2):
	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5, _ = Drawlines(img1,img2,lines1,pts1,pts2)
	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3, _ = Drawlines(img2,img1,lines2,pts2,pts1)
	plt.subplot(121),plt.imshow(img5)
	plt.subplot(122),plt.imshow(img3)
	plt.show()

def main():
		
	x = (np.rint(np.random.rand(10,2)*10))
	xPrime = (np.rint(np.random.rand(10,2)*10))
	
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