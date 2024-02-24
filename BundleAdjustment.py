import numpy as np
import time
from scipy.spatial.transform import Rotation as scipyRot
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from BuildVisibilityMatrix import *
from LinAlgTools import Homogenize

def two_dimension_pts(X_index, visibility_matrix, x_f, y_f):
    
    pts2d = []
    
    x_f = x_f[X_index]
    y_f = y_f[X_index]
    
    for i in range(visibility_matrix.shape[0]):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i][j] == 1:
                pts2d.append(np.hstack)
        pts2d.append()
        
    pts2d = np.array(pts2d).reshape(-1,2)
    return pts2d

def CameraIndices(visibility_matrix):
    cam_indices = []
    point_indices = []
    
    for i in range(visibility_matrix.shape[0]):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i][j] == 1:
                cam_indices.append(j)
                point_indices.append(i)
                
    cam_indices = np.array(cam_indices).reshape(-1)
    point_indices = np.array(point_indices).reshape(-1)
    return cam_indices, point_indices

def bundle_adjustment(X_found, filtered_features, nCam):
    
    X_index, visibility_matrix = VisibilityMatrix(X_found, filtered_features, nCam)
    
    m = np.sum(visibility_matrix) * 2
    n = (nCam+1)*6+X_index[0].shape[0]*3
    A = lil_matrix((m, n), dtype=int)
    
    
    i = np.arrange(np.sum(visibility_matrix))
    
    cam_indices, point_indices = CameraIndices(visibility_matrix)
    
    for s in range(6):
        A[2*i, cam_indices*6+s] = 1
        A[2*i+1, cam_indices*6+s] = 1
        
    for s in range(3):
        A[2*i, (nCam+1)*6+point_indices*3+s] = 1
        A[2*i+1, (nCam+1)*6+point_indices*3+s] = 1
        
    return A

def ProjectPoints(R,C,pts3d, K):
    P = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
    pts3d = Homogenize(pts3d)
    pt_proj = np.dot(P, pts3d.T)
    pt_proj = pt_proj/pt_proj[2,:]
    return pt_proj

def project(pts3d, camera_params, K):
    
    projected_pts = []
    for i in range(len(camera_params)):
        Q = camera_params[i, :3]
        R = scipyRot.from_rotvec(Q).as_matrix()
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = pts3d[i]
        pt_proj = ProjectPoints(R,C,pt3D, K)[:2]
        projected_pts.append(pt_proj)
    return np.array(projected_pts)

def rotate(points, rotation_vectors):
    theta = np.linalg.norm(rotation_vectors, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rotation_vectors / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def Compute_Residual(x0, nCam, n_points, camera_indices, point_indices, points_2d, K):

    number_of_cam = nCam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error = (points_proj - points_2d).ravel()
    
    return error

def BundleAdjustment(X_index,visibility_matrix,X_all,X_found,feature_x,feature_y, filtered_feature_flag, R_set, C_set, K, nCam):
    
    pts_3d = X_all[X_index]
    pts_2d = two_dimension_pts(X_index, visibility_matrix, feature_x, feature_y)
    
    euler_pos = []
    
    for i in range(nCam+1):
        C, R = C_set[i], R_set[i]
        Q = scipyRot.from_matrix(R).as_rotvec()
        pose_ = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        euler_pos.append(pose_)
    
    euler_pos = np.array(euler_pos).reshape(-1,6)
    
    X0 = np.hstack((euler_pos.ravel(), pts_3d.ravel()))
    n_pts = pts_3d.shape[0]
    
    camera_indices, point_indices = CameraIndices(visibility_matrix)
    
    A = bundle_adjustment(X_found, filtered_feature_flag, nCam)
    t0 = time.time()
    res = least_squares(Compute_Residual, X0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(nCam, n_pts, camera_indices, point_indices, pts_2d, K))
    t1 = time.time()
    
    X1 = res.x
    camNum = nCam + 1
    
    optimized_poses = X1[:camNum*6].reshape((camNum, 6))
    optimized_pts3d = X1[camNum*6:].reshape((n_pts, 3))
    optimized_xAll = np.zeros_like(X_all)
    optimized_xAll[X_index] = optimized_pts3d
    
    optimized_Cset, optimized_Rset = [], []
    
    for i in range(len(optimized_poses)):
        R = scipyRot.from_rotvec(optimized_poses[i, :3]).as_matrix()
        C = optimized_poses[i, 3:].reshape(3,1)
        optimized_Cset.append(C)
        optimized_Rset.append(R)
        
    return optimized_Cset, optimized_Rset, optimized_xAll 