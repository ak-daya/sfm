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
    return (R[-1, :] @ (X - C) > 0)