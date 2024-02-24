import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def drawmatches(img1, img2, coordpairs1, coordpairs2):
    keypoints0 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in coordpairs1]
    keypoints1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in coordpairs2]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(coordpairs1))]
    matched_img = cv2.drawMatches(img1, keypoints0, img2, keypoints1, matches, None, flags=2)
    cv2.imshow("Matched Image", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Plot3DPointSets(list_of_3Dpointsets, color_list, legend_list, xlim, ylim, title):
    fig, ax = plt.subplots(1,1)
    # ax = fig.add_subplot(projection='3d')

    for i in range(len(list_of_3Dpointsets)):
        # ax = plt.axes(projection='3d')
        X, Z = list_of_3Dpointsets[i][:, 0], list_of_3Dpointsets[i][:, 2]

        # 3D Plot
        ax.scatter(X, Z, s=1, c = color_list[i], label=legend_list[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    # ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    plt.show()
    

def Plot3DCameraView(points, R_s, C_s, color, legend, xlim, ylim, title):
    fig, ax = plt.subplots(1,1)
    # ax = fig.add_subplot(projection='3d')

    for i in range(len(R_s)):

        r2 = Rotation.from_matrix(R_s[i])
        angles2 = r2.as_euler("zyx", degrees=True)

        plt.plot(C_s[i][0], C_s[i][-1], marker=(3,0,int(angles2[1])), markersize=15, linestyle='None')
        plt.annotate(f"{i+1}", (C_s[i][0], C_s[i][-1]), (C_s[i][0], C_s[i][-1]), ha='center', va='center')
        # ax = plt.axes(projection='3d')
    
    X, Z = points[:, 0], points[:, 2]

    # 3D Plot
    ax.scatter(X, Z, s=1, c = color, label=legend)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    # ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    plt.show()

def Plot3DReconstruction(img, points, fname):
    """
    points: N x 3 (x, y, z) 3d world points
    colors: N x 3 (r, g, b) RGB colors of each image point
    """

    radius = 3
    color = (0, 255, 0)
    thickness = 1

    for center in points:
        cv2.circle(img, center, radius, color, thickness)
    
    cv2.imwrite(f"{fname}.png", img)

    cv2.imshow("Reprojection of estimated 3D points on image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

