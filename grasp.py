import yaml
# from libs.auxiliary import create_folder_with_date, get_ip, popup_message
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from Robotic_Arm import *
from Robotic_Arm.rm_robot_interface import *
# from vertical_grab.convert_eye2hand import convert_new
from vertical_grab.convert_update import convert_new
from cv_process import segment_image
from grasp_process import run_grasp_inference
import time
import open3d as o3d
from config.loader import load_config

# 加载配置参数
config = load_config()

color_intr = {"ppx": config['CAM_INTR']['ppx'], "ppy": config['CAM_INTR']['ppy'], "fx": config['CAM_INTR']['fx'], "fy": config['CAM_INTR']['fy']}
depth_intr = {"ppx": config['DEPTH_INTR']['ppx'], "ppy": config['DEPTH_INTR']['ppy'], "fx": config['DEPTH_INTR']['fx'], "fy": config['DEPTH_INTR']['fy']}

cam_h = config['cam_resolution']['height']
cam_w = config['cam_resolution']['width']

# # 相机内参  640*480
# color_intr = {"ppx": 331.054, "ppy": 240.211, "fx": 604.248, "fy": 604.376}
# depth_intr = {"ppx": 319.304, "ppy": 236.915, "fx": 387.897, "fy": 387.897}


# # 相机内参  1280*720
# color_intr = {"ppx": 656.581, "ppy": 360.316, "fx": 906.373, "fy": 906.563}
# depth_intr = {"ppx": 638.839, "ppy": 354.818, "fx": 646.495, "fy": 646.495}

#手眼标定外参 新：20250316
rotation_matrix = config['rotation_matrix']
translation_vector = config['translation_vector']

# 第一个位置，会使得如果 + 会沿x轴正向移动

# 第二个位置，会使得如果 - 会沿y轴负向移动

# 全局变量
global color_img, depth_img, robot, first_run
color_img = None
depth_img = None
robot = None
first_run = True  # 新增首次运行标志

def get_aligned_frame(self):
        align = rs.align(rs.stream.color)  # type: ignore
        frames = self.pipline.wait_for_frames()
        # aligned_frames 对齐之后结果
        aligned_frames = align.process(frames)
        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        return color, depth

def callback(color_frame, depth_frame):
    global color_img, depth_img
    scaling_factor_x = 1
    scaling_factor_y = 1

    color_img = cv2.resize(
        color_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_AREA
    )
    depth_img = cv2.resize(
        depth_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    if color_img is not None and depth_img is not None:
        test_grasp()

# def pose_to_list(pose):
#     # 从 Pose 对象中提取位置和欧拉角信息
#     x = pose.position.x
#     y = pose.position.y
#     z = pose.position.z
#     rx = pose.euler.rx
#     ry = pose.euler.ry
#     rz = pose.euler.rz
#     return [x, y, z, rx, ry, rz]

def matrix_to_list(x):
    # 从 Pose 对象中提取位置和欧拉角信息
    irow = x.irow
    iline = x.iline
    return np.array(x.data, dtype=np.float32).reshape(irow, iline)

def numpy_to_Matrix(x):
    out = rm_matrix_t()
    out.irow = 4
    out.iline = 4
    out.data = (ctypes.c_float * 16)()
    out.data[:] = x.flatten()
    return out

def test_grasp():
    global color_img, depth_img, robot, first_run, init

    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # 图像处理部分
    masks = segment_image(color_img)  
    
    # translation, rotation_mat_3x3, width = run_grasp_inference(
    #     color_img,
    #     depth_img,
    #     masks
    # )
    predict_result, cloud_o3d = run_grasp_inference(
        color_img,
        depth_img,
        masks
    )
    for i, result in enumerate(predict_result):
        print(f"[DEBUG] 预测结果 {i} ")
        translation, rotation_mat_3x3, width = result.translation, result.rotation_matrix, result.width
        # print(f"[DEBUG] 预测结果 {i+1} - 平移: {translation}, 旋转矩阵:\n{rotation_mat_3x3}")
        error_code, dic_state = robot.rm_get_current_arm_state()
        if not error_code:
            joints = dic_state['joint']
            current_pose = dic_state['pose']
            # print("\n[DEBUG]当前关节角度:", joints)
            # print("\n[DEBUG]未补偿夹爪前位姿:", current_pose_old)
        else:
            print('返回机械臂当前状态失败，检查机械臂连接状况')
            return 

        # print("[DEBUG] 补偿夹爪后的位姿:", current_pose)
        T1 = robot.rm_algo_pos2matrix(current_pose)  # 位姿转换为齐次矩阵
        T_ee2base = matrix_to_list(T1)
        # print("[DEBUG] 官方api计算出对应的齐次矩阵:", T_ee2base)

        T_grasp2base = convert_new(
            translation,
            rotation_mat_3x3,
            rotation_matrix,
            translation_vector,
            T_ee2base
        )
        # print("[DEBUG] 基坐标系抓取齐次矩阵:", T_grasp2base)

        # 判断相机朝向下还是上，防止相机碰到桌面
        T_cam2end = np.eye(4)
        # print('shape of calibration rotaion matrix ', '\n', np.shape(rotation_matrix), type(rotation_matrix))
        # print('shape of calibration translation matrix ', '\n', np.shape(translation_vector), type(translation_vector))
        T_cam2end[:3, :3] = rotation_matrix
        T_cam2end[:3, 3] = translation_vector
        T_cam_pose = T_grasp2base @ T_cam2end
        diff_cam_end = T_cam_pose[2, -1] - T_grasp2base[2, -1]
        matrix_struct = numpy_to_Matrix(T_grasp2base)
        base_pose = robot.rm_algo_matrix2pos(matrix_struct)
        # mode_input = None

        para = rm_inverse_kinematics_params_t(joints, base_pose, 1)
        error_code, joint_new = robot.rm_algo_inverse_kinematics(para)
        if error_code == 0:
            print('逆解成功')
            if diff_cam_end < 0:
                print('相机朝下，可能会怼到桌面，需要调整一下')
                joint_new[-1] += 180
                if joint_new[-1] >= 360:
                    joint_new[-1] -= 360
                elif joint_new[-1] <= -360:
                    joint_new[-1] += 360
                # mode_input = 'joint'
                # base_pose_new = robot.rm_algo_forward_kinematics(joint_new, 1)
            else:
                print('目标位姿相机朝上，暂无碰到桌面的危险')
                # mode_input = 'pose'
        else:
            print('逆解失败，失败码为： ', error_code)
            continue
        

        # 补偿夹爪长度
        base_pose_new = robot.rm_algo_cartesian_tool(joint_new, 0, 0, -0.10)

        # 首次运行只计算不执行
        if first_run:
            print("[INFO] 首次运行模拟完成，准备正式执行")
            first_run = False
            return  # 直接返回不执行后续动作

        try:
            print(f"实际抓取: {base_pose_new}")
            grippers = [g.to_open3d_geometry() for g in [result]]
            o3d.visualization.draw_geometries([cloud_o3d, *grippers])
            input()
            ret = robot.rm_movej_p(base_pose_new, 10, 0, 0, 1)
            if ret != 0: raise RuntimeError(f"抓取失败，错误码: {ret}")

            print("闭合夹爪")
            # ret = robot.Set_Gripper_Pick(200, 300)
            # if ret != 0: raise RuntimeError(f"夹爪闭合失败，错误码: {ret}")

            # robot.Movej_Cmd(init, 10, 0)
            # #robot.Movej_Cmd(fang, 10, 0)
            # robot.Set_Gripper_Release(200)
            time.sleep(5)
            robot.rm_movej_p(init, 20, 0, 0, 1)
        except Exception as e:
            print(f"[ERROR] 运动异常: {str(e)}")
            robot.rm_movej_p(init, 20, 0, 0, 1)
        
        if ret == 0:
            print(f"[INFO] 运动成功")
            break

    # print(f"[DEBUG] Grasp预测结果 - 平移: {translation}, 旋转矩阵:\n{rotation_mat_3x3}")

    # error_code, dic_state = robot.rm_get_current_arm_state()
    # if not error_code:
    #     joints = dic_state['joint']
    #     current_pose_old = dic_state['pose']
    #     print("\n[DEBUG]当前关节角度:", joints)
    #     print("\n[DEBUG]未补偿夹爪前位姿:", current_pose_old)
    # else:
    #     print('返回机械臂当前状态失败，检查机械臂连接状况')
    #     return 


    # # 补偿夹爪长度
    # current_pose = robot.rm_algo_cartesian_tool(joints, 0, 0, -0.05)
    # print("[DEBUG] 补偿夹爪后的位姿:", current_pose)
    # T1 = robot.rm_algo_pos2matrix(current_pose)  # 位姿转换为齐次矩阵
    # T_ee2base = matrix_to_list(T1)
    # # print("[DEBUG] 官方api计算出对应的齐次矩阵:", T_ee2base)

    # T_grasp2base = convert_new(
    #     translation,
    #     rotation_mat_3x3,
    #     rotation_matrix,
    #     translation_vector,
    #     T_ee2base
    # )
    # print("[DEBUG] 基坐标系抓取齐次矩阵:", T_grasp2base)

    # # 判断相机朝向下还是上，防止相机碰到桌面
    # T_cam2end = np.eye(4)
    # print('shape of calibration rotaion matrix ', '\n', np.shape(rotation_matrix), type(rotation_matrix))
    # print('shape of calibration translation matrix ', '\n', np.shape(translation_vector), type(translation_vector))
    # T_cam2end[:3, :3] = rotation_matrix
    # T_cam2end[:3, 3] = translation_vector
    # T_cam_pose = T_grasp2base @ T_cam2end
    # diff_cam_end = T_cam_pose[2, -1] - T_grasp2base[2, -1]
    # matrix_struct = numpy_to_Matrix(T_grasp2base)
    # base_pose = robot.rm_algo_matrix2pos(matrix_struct)
    # mode_input = None
    # if diff_cam_end < 0:
    #     print('相机朝下，可能会怼到桌面，需要调整一下')
    #     para = rm_inverse_kinematics_params_t(joints, base_pose, 1)
    #     error_code, joint_new = robot.rm_algo_inverse_kinematics(para)
    #     if error_code == 0:
    #         print('逆解成功')
    #         joint_new[-1] += 180
    #         mode_input = 'joint'
    #     else:
    #         print('逆解失败，失败码为： ', error_code)
    # else:
    #     print('目标位姿相机朝上，暂无碰到桌面的危险')
    #     mode_input = 'pose'

    # print("[DEBUG] 最终抓取位姿是什么:", base_pose)

    # # 首次运行只计算不执行
    # if first_run:
    #     print("[INFO] 首次运行模拟完成，准备正式执行")
    #     first_run = False
    #     return  # 直接返回不执行后续动作

    # # 正式执行部分
    # # base_pose_np = np.array(base_pose, dtype=float)
    # # base_xyz = base_pose_np[:3]
    # # base_rxyz = base_pose_np[3:]

    # # # 预抓取计算
    # # pre_grasp_offset = 0.1
    # # pre_grasp_pose = np.array(base_pose, dtype=float).copy()
    # # # rotation_mat = R.from_euler('xyz', pre_grasp_pose[3:]).as_matrix()
    # # # z_axis = rotation_mat[:, 2]
    # # # pre_grasp_pose[:3] -= z_axis * pre_grasp_offset

    # # rotation_mat = np.array(T_grasp2base[:3, :3], dtype=float)
    # # z_axis = rotation_mat[:, 2]
    # # pre_grasp_pose[:3] -= z_axis * pre_grasp_offset

    # # fang = [-20, 25, 0, -90, 0, 25, 0]

    # try:
    #     # input()
    #     # print(f"预抓取位姿: {pre_grasp_pose.tolist()}")
    #     # ret = robot.rm_movej_p(pre_grasp_pose, 15, 0, 0, 1)
    #     # if ret != 0: raise RuntimeError(f"预抓取失败，错误码: {ret}")
        

    #     print(f"实际抓取: {base_pose}")
    #     input()
    #     if mode_input == 'joint':
    #         ret = robot.rm_movej(joint_new, 10, 0, 0, 1)
    #     else:
    #         ret = robot.rm_movej_p(base_pose, 15, 0, 0, 1)
    #     if ret != 0: raise RuntimeError(f"抓取失败，错误码: {ret}")

    #     print("闭合夹爪")
    #     # ret = robot.Set_Gripper_Pick(200, 300)
    #     # if ret != 0: raise RuntimeError(f"夹爪闭合失败，错误码: {ret}")

    #     # robot.Movej_Cmd(init, 10, 0)
    #     # #robot.Movej_Cmd(fang, 10, 0)
    #     # robot.Set_Gripper_Release(200)
    #     time.sleep(5)
    #     robot.rm_movej_p(init, 20, 0, 0, 1)
    # except Exception as e:
    #     print(f"[ERROR] 运动异常: {str(e)}")
    #     robot.rm_movej_p(init, 20, 0, 0, 1)



def displayD435():
    global first_run, init, robot

    robot.rm_movej_p(init, 10, 0, 0, 1)
    pipeline = rs.pipeline()
    config = rs.config()
    time.sleep(3)
    config.enable_stream(rs.stream.color, cam_w, cam_h, rs.format.bgr8, 6)
    config.enable_stream(rs.stream.depth, cam_w, cam_h, rs.format.z16, 6)

    try:
        robot.rm_movej_p(init, 10, 0, 0, 1)
        print('connect to camera')
        profile = pipeline.start(config)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        # 新增：创建对齐对象，将深度图与彩色图对齐
        align = rs.align(rs.stream.color)  # 对齐到彩色图像流

        while True:
            frames = pipeline.wait_for_frames()
            if not frames:
                continue

            # 对齐帧
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            callback(color_image, depth_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    global robot, first_run, arm, init
    robot_ip = config['robot_ip']
    # logger_.info(f'robot_ip:{robot_ip}')

    if robot_ip:
        robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        arm = robot.rm_create_robot_arm(robot_ip, 8080)
        # print(robot.API_Version())
    else:
        print("提醒", "机械臂 IP 没有 ping 通")
        sys.exit(1)

    # 初始化设置
    init = [-0.20807, 0.000295, 0.488525, 3.142, 0.326, 0.001]
    robot.rm_movej_p(init, 10, 0, 0, 1)

    # 重置首次运行标志
    first_run = True
    displayD435()


if __name__ == "__main__":
    def get_aligned_frame(self):
        align = rs.align(rs.stream.color)  # type: ignore
        frames = self.pipline.wait_for_frames()
        # aligned_frames 对齐之后结果
        aligned_frames = align.process(frames)
        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        return color, depth
    main()


    