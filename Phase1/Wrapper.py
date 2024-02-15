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
    for index, line in enumerate(file):
        if index > 0 :
            numbers = line.split()
            numbers = [float(num) for num in numbers]
            for j in range(int(numbers[0])-1):
                ImgPairIndex = 3*j+6
                ImgPair = int(numbers[ImgPairIndex])
                pairs[ImgPair-(imgindex+1)].append([[numbers[ImgPairIndex-2], numbers[ImgPairIndex-1]],[numbers[ImgPairIndex+1], numbers[ImgPairIndex+2]]])
        else:
            continue
    return pairs

def returnpairs(matchingpairs, pair):
    pair = sorted(pair)
    return matchingpairs[pair[0]-1][pair[1]-pair[0]-1]
            
    


def main():
    basePath = dirname(dirname(abspath(__file__)))
    image_path = basePath + f"\\sfmdata"
    txtdata_path = basePath + f"\\sfmtxtdata"
    
    images = LoadImagesFromFolder(image_path)
    intrinsic_matrix = np.array([[531.122155322710, 0.0, 407.192550839899], [0.0, 531.541737503901, 313.308715048366], [0.0, 0.0, 1.0]])
    MatchPairs_text = LoadTextFromFolder(txtdata_path)
    Matchpairs = []
    for index, file in enumerate(MatchPairs_text):
        ImgPairs = Matching_pairs(file, index+1)
        Matchpairs.append(ImgPairs)
    print(returnpairs(Matchpairs, [3,1]))
    
    
    
    
if __name__ == "__main__":
	main()