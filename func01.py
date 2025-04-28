import cv2
import numpy as np

def pixel_to_camera_coord(u, v, depth_value, K, D):
    # 去畸变
    distorted_pt = np.array([[u, v]], dtype=np.float32)
    undistorted_pt = cv2.undistortPoints(distorted_pt, K, D, P=K)
    u_undistorted, v_undistorted = undistorted_pt[0][0]
    
    # 归一化坐标
    z = 1.0
    x_norm = (u_undistorted - K[0, 2]) * z / K[0, 0]
    y_norm = (v_undistorted - K[1, 2]) * z / K[1, 1]
    
    # 转换为相机坐标系
    Xc = x_norm * depth_value
    Yc = y_norm * depth_value
    Zc = depth_value
    
    return Xc, Yc, Zc
