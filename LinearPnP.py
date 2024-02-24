import numpy as np

def linear_pnp(K, pts2d, pts3d):
    
    zeros, ones = np.zeros((pts3d.shape[0])), np.ones((pts3d.shape[0]))
    
    x, y, z = pts3d[:,0], pts3d[:,1], pts3d[:,2]
    u, v = pts2d[:,0], pts2d[:,1]
    A1 = np.vstack([x, y, z, ones, zeros, zeros, zeros, zeros, -u*x, -u*y, -u*z, -u]).T
    A2 = np.vstack([zeros, zeros, zeros, zeros, x, y, z, ones, -v*x, -v*y, -v*z, -v]).T
    A = np.vstack([A1, A2])
    U, S, Vt = np.linalg.svd(A)
    P = Vt[np.argmin(S), :].reshape(3,4)
    R = np.linalg.inv(K) @ P[:, :3]
    Ur, Sr, Vtr = np.linalg.svd(R)
    R = Ur @ Vtr
    R_det = np.linalg.det(R)
    T = np.linalg.inv(K) @ P[:, 3]
    if R_det < 0:
        R = -R
        T = -T 
    C = -R.T @ T
    return R, C