import cv2
from os import listdir

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