import numpy as np
from LinAlgTools import Homogenize
from LinearPnP import linear_pnp


def PnpRansac(K, pts2d, pts3d, n_iters=1000, threshold = 1e-10):
    H_pts3d = Homogenize(pts3d)
    u, v = pts2d[:,0], pts2d[:,1]
    max_inliers = []
    for iter in range(n_iters):
        rnd_idx = np.random.choice(pts3d.shape[0], 6, replace=False)
        rnd_pts2d = pts2d[rnd_idx]
        rnd_pts3d = H_pts3d[rnd_idx]
        
        R, C = linear_pnp(K, rnd_pts2d, rnd_pts3d)
        
        T = -R@C.reshape((3,1))#Translation vector from world to camera
        P = K @ np.concatenate([R, T], axis=1)
        
        u_ = np.dot(P[0], H_pts3d.T)/np.dot(P[2], H_pts3d.T)
        v_ = np.dot(P[1], H_pts3d.T)/np.dot(P[2], H_pts3d.T)
        
        e = (u- u_)**2 + (v - v_)**2
        
        curr_inliers = abs(e) < threshold
        
        if len(curr_inliers) > len(max_inliers):
            max_inliers = curr_inliers
            
    pts2d_inliers = pts2d[max_inliers]
    pts3d_inliers = pts3d[max_inliers]
    
    R,C = linear_pnp(K, pts2d_inliers, pts3d_inliers)
    
    return R, C, pts2d_inliers, pts3d_inliers