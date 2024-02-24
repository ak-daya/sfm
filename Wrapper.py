import numpy as np
import cv2
import matplotlib.pyplot as plt 
import copy
import os
import scipy.optimize as optimize
import math
from os.path import dirname, abspath
from skimage.io import imread, imshow
from skimage import transform
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from EstimateFundamentalMatrix import *
from GetInlierRANSAC import *
from DisambiguateCameraPose import *
from rotationmatrix import *
from Visualization import *
from DataLoader import *   
from LinearTriangulation import *   
from NonlinearTriangulation import *
from PnPRANSAC import *

def main():
    # Read in images, correspondences, intrinsic matrix
    basePath = (dirname(abspath(__file__)))
    image_path = basePath + f"\\Data\\sfmdata"
    txtdata_path = basePath + f"\\Data\\sfmtxtdata"
    
    images = LoadImagesFromFolder(image_path)
    K = np.array([[531.122155322710, 0.0, 407.192550839899], [0.0, 531.541737503901, 313.308715048366], [0.0, 0.0, 1.0]])
    MatchPairs_text = LoadTextFromFolder(txtdata_path)
    Matchpairs = []
    Colorpairs = []
    for index, file in enumerate(MatchPairs_text):
        ImgPairs, colorpairs = Matching_pairs(file, index+1)
        Matchpairs.append(ImgPairs)
        Colorpairs.append(colorpairs)
    
    # print(returnpairs(Matchpairs, [3,1]))
    # FundamentalMatrix = np.array([[0.1, 0.2, -0.3], [0.4, 0.5, -0.6], [0.7, 0.8, -0.9]])
    # essentialMatrix = EssentialMatrix(intrinsic_matrix, FundamentalMatrix)
    # print(essentialMatrix)
    # pose = CameraPoseEstimation(essentialMatrix)
    # print(pose)
        
    # Outlier rejection for all pairs
    
    
    # First two images
    coord_pair = np.array(returnpairs(Matchpairs, [1,2]))
    coordpairs1 = coord_pair[:,0,:] #[pair[0] for pair in coord_pair]
    coordpairs2 = coord_pair[:,1,:] #[pair[0] for pair in coord_pair]
    
    # Debug
    # print(f"Number of matches: {len(coordpairs1)}")
    
    # best_points1H, best_points2H = HomographyRansac(coord_pair)
    # drawmatches(images[i], images[j], coordpairs1, coordpairs2)
    # drawmatches(images[i], images[j], best_points1H, best_points2H)

    best_points1, best_points2 = OutlierRejectionRANSAC(coordpairs1, coordpairs2, break_percentage=0.9)
    # best_points1, best_points2 = OutlierRejectionRANSAC(np.array(best_points1H), np.array(best_points2H))

    # Debug
    print(f"Number of pruned matches: {len(best_points1)}")

    # Visualize sample of best matches
    # rand_idx = random.sample(range(best_points1.shape[0]), 16)
    # coordpairs1_sample = best_points1[rand_idx]
    # coordpairs2_sample = best_points2[rand_idx]
    # drawmatches(images[i], images[j], coordpairs1_sample, coordpairs2_sample)
    
    # Estimate fundamental matrix over best matches
    fundamentalMatrix = EstimateFundamentalMatrix(best_points1, best_points2, normalize=False)
    # print(f"F: \n{fundamentalMatrix}")
    
    # Estimate essential matrix            
    essentialMatrix = EssentialMatrixFromFundamentalMatrix(K, fundamentalMatrix)
    # print(f"F: \n{essentialMatrix}")

    # Testing for epipolar constraint
    # best_points1_hom = Homogenize(best_points1)
    # best_points2_hom = Homogenize(best_points2)
    # epiConstraint1 = 0
    # epiConstraint2 = 0
    # for i in range(best_points1_hom.shape[0]):
    #     epiConstraint1 += best_points2_hom[i] @ Fundamental_matrix @ best_points1_hom[i]
    #     epiConstraint2 += best_points2_hom[i] @ FundamentalMatrixCV2 @ best_points1_hom[i]
    # avgEpiConstraint1 = epiConstraint1/best_points1_hom.shape[0]
    # avgEpiConstraint2 = epiConstraint2/best_points1_hom.shape[0]
    # print(f"Avg Epipolar Constraint Ours: {avgEpiConstraint1}")
    # print(f"Avg Epipolar Constraint CV2: {avgEpiConstraint2}")
    
    # Estimate camera poses
    C_s, R_s = ExtractCameraPose(essentialMatrix)
    
    # Linear Triangulation
    # Iterate over poses, estimate depth and do cheirality check
    C_O = np.zeros((3,1))
    R_O = np.identity(3)

    linearDepthPts = []
    for i in range(len(R_s)):
        points = LinearTriangulation(K, C_O, R_O, C_s[i], R_s[i], best_points1, best_points2)
        linearDepthPts.append(points)
    
    # Visualize all poses
    Plot3DPointSets(linearDepthPts, ['brown','blue','pink','purple'], ['Pose 1', 'Pose 2', 'Pose 3', 'Pose 4'], 
                    [-30, 30], [-30, 30], 'Triangulation over All Possible Poses')

    # Check cheirality condition
    C, R, linearDepth = DisambiguateCameraPose(C_s, R_s, linearDepthPts)

    # Non-linear Triangulation
    X0 = linearDepth

    nonlinearDepth = NonLinearTriangulation(X0, K, C_O, R_O, C, R, best_points1, best_points2)

    # Visualize Linear and Non-Linear Triangulation
    Plot3DPointSets([X0, nonlinearDepth], ['blue','red'], ['Linear', 'Non-Linear'], 
                    [-20, 20], [-5, 30], 'Linear and Non-Linear Triangulation')
    
    
    P1 = GetProjectionMatrix(C_O, R_O, K)
    P2 = GetProjectionMatrix(C, R, K)
    P = [P1, P2]

    viz_linear = ["Linear_Tri_Img1","Linear_Tri_Img2"]
    viz_nonlinear = ["nonLinear_Tri_Img1","nonLinear_Tri_Img2"]

    for i in range(2):
        img = images[i]
        
        linear_tri_uv = World2Image(linearDepth, P[i])
        nonlinear_tri_uv = World2Image(nonlinearDepth, P[i])

        linear_tri_uv = np.rint(linear_tri_uv).astype(np.uint16)
        nonlinear_tri_uv = np.rint(nonlinear_tri_uv).astype(np.uint16)

        Plot3DReconstruction(img, linear_tri_uv, viz_linear[i])
        Plot3DReconstruction(img, nonlinear_tri_uv, viz_nonlinear[i])

    Plot3DCameraView(nonlinearDepth, [R], [C], 'blue', '', [-20, 20], [-5, 30], "Estimated 3D Points and Camera Pose")

    # 
    # print((X0 - nonlinearDepthPts)[:5])
    # print(np.linalg.norm(X0 - nonlinearDepthPts))
    
    
    # Visualizer4R(depthpoints)
    # visualizer(depthpointsCheralityCheck[max_index])

    #non-linear triangulation
    # pointsNonLinear = TriangulateDepth_NonLinear([intrinsic_matrix, C1, R1, C_s[max_index], R_s[max_index]], best_points1, best_points2, pointsLinear)
    # visualizer(pointsNonLinear)

    # best_pts3d = depthpointsCheralityCheck[max_index]

    
    # Images 2-5
    
    
if __name__ == "__main__":
	main()