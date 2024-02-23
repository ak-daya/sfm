from LinearTriangulation import TriangulateDepth_Linear

def CheiralityCondition(C, R, X):
    """
    Computes whether the world point X lies in front of
    the camera's z-axis whose pose is given by 
    camera center C, and rotation R

    C: Camera position
    R: Camera rotation
    X: World point w.r.t. some camera coordinate system
    """
    condition_1 = R[-1, :] @ (X - C).T > 0
    condition_2 = X[2] > 0
    Main_condition = condition_1 and condition_2
    return Main_condition