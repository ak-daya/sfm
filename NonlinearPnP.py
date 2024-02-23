import numpy as np
from EstimateFundamentalMatrix import Homogenize
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as scipyRot
import cv2

def loss_fn(X0, pts3d, pts2d, K):
    Q, C = X0[:4], X0[4:].reshape(-1,1)
    R = scipyRot.from_quat(Q).as_matrix()
    I = np.identity(3)
    C = np.reshape(C, (3,1))
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    
    p_1T, p_2T, p_3T = P[0].reshape(1,-1), P[1].reshape(1,-1), P[2].reshape(1,-1)
    
    E = []
     
    for i in range(len(pts2d)):
        p_3d = Homogenize(pts3d[i])
        u,v = pts2d[i][0], pts2d[i][1]
        p_2d = np.dot(P, p_3d)
        p_2d = p_2d/p_2d[-1]
        u_proj = p_2d[0]
        v_proj = p_2d[1]
        e = np.square(v - v_proj) + np.square(u - u_proj)
        
        E.append(e)
        
    ErrorAvg = np.mean(np.array(E).squeeze())
    return ErrorAvg

def NonlinearPnP(K, pts2d, pts3d, R, C):
    Q = scipyRot.from_matrix(R).as_quat()
    
    X0 = [Q[0] ,Q[1],Q[2],Q[3], C[0], C[1], C[2]]
    
    optimized_params = least_squares(
                        fun = loss_fn,
                        x0=X0,
                        method="trf",
                        args=[pts3d, pts2d, K])
    
    Q1 = optimized_params.x[:4]
    C1 = optimized_params.x[4:]
    R1 = scipyRot.from_quat(Q1).as_matrix()
    
    return R1, C1