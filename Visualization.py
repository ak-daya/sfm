import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Plot3DPointSets(list_of_3Dpointsets, color_list, legend_list):
    fig, ax = plt.subplots(1,1)
    # ax = fig.add_subplot(projection='3d')

    for i in range(len(list_of_3Dpointsets)):
        # ax = plt.axes(projection='3d')
        X, Z = list_of_3Dpointsets[i][:, 0], list_of_3Dpointsets[i][:, 2]

        # 3D Plot
        ax.scatter(X, Z, s=1, c = color_list[i], label=legend_list[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    # ax.set_zlabel('Z')
    ax.set_title('Triangulation over All Possible Poses')
    ax.legend()

    plt.show()
    

def Plot3DReconstruction(points, colors_rgb):
    """
    points: N x 3 (x, y, z) 3d world points
    colors: N x 3 (r, g, b) RGB colors of each image point
    """
    # ax = plt.axes(projection='3d')
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c = colors_rgb)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction of Scene')

    plt.show()

