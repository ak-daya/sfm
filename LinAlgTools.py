import numpy as np

def Skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def Homogenize(coordinates):
	# Adds a dimension of 1
	if len(coordinates.shape) == 1:
		hom_coordinates = np.ones((coordinates.shape[0] + 1))
		hom_coordinates[:-1] = coordinates
	else:
		hom_coordinates = np.ones((coordinates.shape[0], coordinates.shape[1]+1))
		hom_coordinates[:, :-1] = coordinates
	return hom_coordinates