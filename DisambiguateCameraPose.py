import numpy as np

def DisambiguateCameraPose(C_set, R_set, Depth_set):
    count = []
    for i in range(4):
        points = Depth_set[i]
        C = C_set[i]
        C = C.reshape(3,)
        R = R_set[i]

        # Cheirality condition
        diff = points - C
        cheirality = diff @ R[:,-1]
        z_col = points[:,-1]

        condition_1 = cheirality > 0
        condition_2 = z_col > 0

        count.append((condition_1 * condition_2).sum())

    best_idx = np.argmax(count)
    best_C = C_set[best_idx]
    best_R = R_set[best_idx]
    best_depth = Depth_set[best_idx]
    
    return best_C, best_R, best_depth
    