import numpy as np


def VisibilityMatrix(X, filtered_features, nCam):
    
    bin_temp = np.zeros((filtered_features.shape[0]), dtype = int)
    for n in range(nCam + 1):
        bin_temp = bin_temp | filtered_features[:,n]

    X_index = np.where((X.reshape(-1)) & (bin_temp))
    
    visiblity_matrix = X[X_index].reshape(-1,1)
    for n in range(nCam + 1):
        visiblity_matrix = np.hstack((visiblity_matrix, filtered_features[X_index, n].reshape(-1,1)))
    
    return X_index, visiblity_matrix[:, 1:visiblity_matrix.shape[1]]