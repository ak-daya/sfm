import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a cube
def plot_rotation(transformation_matrix):
    cube_vertices = np.array([[0, 0, 0],
                            [1, 0, 0],
                            [1, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 1],
                            [1, 1, 1],
                            [0, 1, 1]])

    # Define transformation matrix (example)
    # This transformation rotates the cube around the x-axis by 45 degrees and translates it
    # You would replace this with your actual transformation matrix
    # transformation_matrix = np.array([[1, 0, 0, 0],   # Rotation part
    #                                 [0, np.cos(np.pi/4), -np.sin(np.pi/4), 0],
    #                                 [0, np.sin(np.pi/4), np.cos(np.pi/4), 0],
    #                                 [0, 0, 0, 1]])  # Translation part

    # Apply transformation to cube vertices
    transformed_cube = np.dot(transformation_matrix[:3, :3], cube_vertices.T).T + transformation_matrix[:3, 3]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original cube
    ax.scatter(cube_vertices[:, 0], cube_vertices[:, 1], cube_vertices[:, 2], color='blue')
    for i in range(4):
        ax.plot3D([cube_vertices[i, 0], cube_vertices[(i + 1) % 4, 0]],
                [cube_vertices[i, 1], cube_vertices[(i + 1) % 4, 1]],
                [cube_vertices[i, 2], cube_vertices[(i + 1) % 4, 2]], color='blue')
        ax.plot3D([cube_vertices[i + 4, 0], cube_vertices[(i + 1) % 4 + 4, 0]],
                [cube_vertices[i + 4, 1], cube_vertices[(i + 1) % 4 + 4, 1]],
                [cube_vertices[i + 4, 2], cube_vertices[(i + 1) % 4 + 4, 2]], color='blue')
        ax.plot3D([cube_vertices[i, 0], cube_vertices[i + 4, 0]],
                [cube_vertices[i, 1], cube_vertices[i + 4, 1]],
                [cube_vertices[i, 2], cube_vertices[i + 4, 2]], color='blue')

    # Plot transformed cube
    ax.scatter(transformed_cube[:, 0], transformed_cube[:, 1], transformed_cube[:, 2], color='red')
    for i in range(4):
        ax.plot3D([transformed_cube[i, 0], transformed_cube[(i + 1) % 4, 0]],
                [transformed_cube[i, 1], transformed_cube[(i + 1) % 4, 1]],
                [transformed_cube[i, 2], transformed_cube[(i + 1) % 4, 2]], color='red')
        ax.plot3D([transformed_cube[i + 4, 0], transformed_cube[(i + 1) % 4 + 4, 0]],
                [transformed_cube[i + 4, 1], transformed_cube[(i + 1) % 4 + 4, 1]],
                [transformed_cube[i + 4, 2], transformed_cube[(i + 1) % 4 + 4, 2]], color='red')
        ax.plot3D([transformed_cube[i, 0], transformed_cube[i + 4, 0]],
                [transformed_cube[i, 1], transformed_cube[i + 4, 1]],
                [transformed_cube[i, 2], transformed_cube[i + 4, 2]], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Transformation Visualization')
    plt.show()