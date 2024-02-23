import numpy as np
import cv2
import matplotlib.pyplot as plt 
import copy
import os
import scipy.optimize as optimize
import math
from os import listdir
from os.path import dirname, abspath
from skimage.io import imread, imshow
from skimage import transform
from EssentialMatrixFromFundamentalMatrix import EssentialMatrix
from ExtractCameraPose import CameraPoseEstimation
from EstimateFundamentalMatrix import *
from GetInlierRANSAC import *
from DisambiguateCameraPose import *
from rotationmatrix import *
from Visualization import *

def LoadImagesFromFolder(folder):
	images = []
	for file in listdir(folder):
		tmp = cv2.imread(folder + "\\" + file)
		if tmp is not None:
			images.append(tmp)
	return images

def LoadTextFromFolder(folder):
    texts = []
    for file in listdir(folder):
        tmp = open(folder + "\\" + file, "r")
        if tmp is not None:
            texts.append(tmp)
    return texts

def Matching_pairs(file, imgindex):
    length = 5-imgindex
    pairs = [[] for _ in range(length)]
    colors = [[] for _ in range(length)]
    for index, line in enumerate(file):
        if index > 0 :
            numbers = line.split()
            numbers = [float(num) for num in numbers]
            for j in range(int(numbers[0])-1):
                ImgPairIndex = 3*j+6
                ImgPair = int(numbers[ImgPairIndex])
                pairs[ImgPair-(imgindex+1)].append([[numbers[ImgPairIndex-2], numbers[ImgPairIndex-1]],[numbers[ImgPairIndex+1], numbers[ImgPairIndex+2]]])
                colors[ImgPair-(imgindex+1)].append([numbers[1],numbers[2], numbers[3]])
        else:
            continue
    return pairs, colors

def returnpairs(matchingpairs, pair):
    pair = sorted(pair)
    return matchingpairs[pair[0]-1][pair[1]-pair[0]-1]

def drawmatches(img1, img2, coordpairs1, coordpairs2):
    keypoints0 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in coordpairs1]
    keypoints1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in coordpairs2]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(coordpairs1))]
    matched_img = cv2.drawMatches(img1, keypoints0, img2, keypoints1, matches, None, flags=2)
    cv2.imshow("Matched Image", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            
    


def main():
    # Read in images, correspondences, intrinsic matrix
    basePath = (dirname(abspath(__file__)))
    image_path = basePath + f"\\Data\\sfmdata"
    txtdata_path = basePath + f"\\Data\\sfmtxtdata"
    
    images = LoadImagesFromFolder(image_path)
    intrinsic_matrix = np.array([[531.122155322710, 0.0, 407.192550839899], [0.0, 531.541737503901, 313.308715048366], [0.0, 0.0, 1.0]])
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
    
    for i in range (0,len(images)-1):
        for j in range(1,len(images)):
            coord_pair = returnpairs(Matchpairs, [i+1,j+1])
            coordpairs1 = [pair[0] for pair in coord_pair]
            coordpairs1 = np.array(coordpairs1)
            coordpairs2 = [pair[1] for pair in coord_pair]
            coordpairs2 = np.array(coordpairs2)
            
            # Debug
            # print(f"Number of matches: {len(coordpairs1)}")
            
            best_points1, best_points2 = OutlierRejectionRANSAC(np.array(coordpairs1), np.array(coordpairs2), break_percentage=0.99)
            
            # Debug
            print(f"Number of pruned matches: {len(best_points1)}")

            # Visualize sample of best matches
            # rand_idx = random.sample(range(best_points1.shape[0]), 16)
            # coordpairs1_sample = best_points1[rand_idx]
            # coordpairs2_sample = best_points2[rand_idx]
            # drawmatches(images[i], images[j], coordpairs1_sample, coordpairs2_sample)
            
            # Compute fundamnetal matrix over best matches
            Fundamental_matrix = EstimateFundamentalMatrix(best_points1, best_points2, normalize=True)
            # print("F= \n",Fundamental_matrix)
            # FundamentalMatrixCV2, _ = cv2.findFundamentalMat(best_points1, best_points2, cv2.FM_RANSAC)
            
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
            
            essentialMatrix = EssentialMatrix(intrinsic_matrix, Fundamental_matrix)
            # print("E: \n", essentialMatrix)

            R_s, C_s = CameraPoseEstimation(essentialMatrix)
            
            RealPose = np.zeros((4,4))
            
            # Relative left camera pose at origin
            R_origin = np.identity(3)
            C_origin = np.array([0,0,0])
            
            depthlen = []
            depthpoints =[[] for _ in range(4)]

            all_depth_points = []

            for i in range(len(R_s)):
                points = TriangulateDepth_Linear(C_s[i], R_s[i], intrinsic_matrix, best_points1, best_points2)

                # For plotting
                all_depth_points.append(points)

                # Cheirality check
                for point in points:
                    if CheiralityCondition(C_s[i], R_s[i], point):
                        depthpoints[i].append(point)

                # Keep count of cheirality condition check
                depthlen.append(len(depthpoints[i]))
            
            # Visualize points
            Plot3DPointSets(all_depth_points, ['brown','blue','pink','purple'], 
                            ['Pose 1', 'Pose 2', 'Pose 3', 'Pose 4'])

            max_index = depthlen.index(max(depthlen))
            RealPose[0:3,0:3] = R_s[max_index]
            RealPose[0:3,3] = C_s[max_index]
            RealPose[3,3] = 1
            plot_rotation(RealPose)
            depthpoints = np.array(depthpoints[max_index])
            
            print(depthpoints)
            break
        break
            
            
        
    
    
    
    
    
if __name__ == "__main__":
	main()