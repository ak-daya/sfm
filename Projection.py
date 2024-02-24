from LinAlgTools import *
import numpy as np

def GetProjectionMatrix(C, R, K):
    # Compute P from camera parameters
    P = K @ R @ np.concatenate([np.identity(3), -C.reshape((3,1))], axis=1)

    return P

def World2Image(WorldPts, P):
    # Project and normalize
    u_pred = Homogenize(WorldPts) @ P.T
    last_elements = u_pred[:,-1]
    u_pred = u_pred / last_elements[:, None]
    
    # Ignore homogeneous coordinate
    u_pred = u_pred[:,:-1]

    return u_pred

def ReprojectionError(u_pred, u_observed):
    error = np.sum((u_observed - u_pred)**2, axis=1)

    return error