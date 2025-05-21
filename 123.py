import time
import threading
import os
import numpy as np
from Robotic_Arm import *
from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs
import cv2

# 系统配置
MASTER_IP = '192.168.110.119'  # 主臂IP
SLAVE_IP = '192.168.110.118'   # 从臂IP
TCP_PORT = 8080
MIN_MOVEMENT_THRESHOLD = 0.05  # 最小运动阈值(弧度)
last_sent_joint = None  # 上次发送的关节位置
color_image_1 = None  # 用于保存图像

# 初始化主臂和从臂
master_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
slave_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
master_handle = master_arm.rm_create_robot_arm(MASTER_IP, TCP_PORT)
slave_handle = slave_arm.rm_create_robot_arm(SLAVE_IP, TCP_PORT)

file_name = '123.txt'
data_path = 'C:/Users/admin/Desktop/internship/try/graspnet_rm65_poplar/data_image/a03'
folder_path = os.path.join(data_path, 'images')
# 确保目录存在
os.makedirs(folder_path, exist_ok=True)
txt_path = os.path.join(data_path, file_name)

# 确保文件存在，否则创建空文件
if not os.path.exists(txt_path):
    with open(txt_path, 'w', encoding='utf-8') as f:
        pass

save_trajectory = []
save_images = []

# 全局锁
lock = threading.Lock()

def get_master_joints():
    """获取主臂当前关节位置"""
    master_status = master_arm.rm_get_current_arm_state()
    if master_status[0] != 0:
        print("无法获取主臂当前状态，错误码:", master_status[0])
        return None
    
    return master_status[1]['joint']

def main_loop():
    """主循环函数"""
    global last_sent_joint, save_trajectory, save_images, color_image_1
    
    try:
        print("启动主从臂实时控制...")
        
        # 确保从臂处于正常状态
        slave_status = slave_arm.rm_get_current_arm_state()
        print(f"从臂状态: {slave_status}")
        if slave_status[0] != 0:
            print("从臂未处于就绪状态，尝试复位...")
            slave_arm.rm_reset_error()
        
        # 打印主臂初始位置
        master_joints = get_master_joints()
        if master_joints:
            print(f"主臂初始关节位置(弧度): {master_joints}")
        
        # 初始化管道和配置
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB流配置

        # 启动流
        profile = pipeline.start(config)
        # frames = pipeline.wait_for_frames()
          
        # 主循环
        while True:
            # 记录循环开始时间
            start_time = time.time()
            
            # 获取主臂关节位置
            master_joints = get_master_joints()
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            
            if master_joints:
                # print(f"主臂当前关节位置: {master_joints}")
                
                # 只有当移动距离超过阈值时才发送命令
                if last_sent_joint is None or np.linalg.norm(np.array(master_joints) - np.array(last_sent_joint)) > MIN_MOVEMENT_THRESHOLD:
                    # print(f"发送关节位置到从臂: {master_joints}")

                    # frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        print("无法获取颜色帧")
                        continue
                    color_image = np.asanyarray(color_frame.get_data())

                    # 直接使用主臂关节位置控制从臂
                    result = slave_arm.rm_movej_follow(master_joints)
                    
                    if result != 0:
                        print(f"移动指令失败，错误码: {result}，尝试复位...")
                        slave_arm.rm_reset_error()
                    else:
                        last_sent_joint = master_joints.copy()
                        color_image_1 = color_image.copy()
                        save_trajectory.append(master_joints)
                        save_images.append(color_image_1)
                else:
                    print("变化量小于阈值，跳过")
            else:
                print('无法获取主臂数据') 
            
            # 计算已用时间
            elapsed_time = time.time() - start_time
            # 动态调整等待时间，确保每次循环大约0.1秒
            sleep_time = max(0, 0.005 - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            # 打印实际循环时间
            print(f"循环时间: {time.time() - start_time:.3f}秒")

    except KeyboardInterrupt:
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            for ii, data in enumerate(save_trajectory):
                input_data = [ii] + data
                f.write(f"{input_data}\n")
        print("保存轨迹数据到文件:", txt_path)
        
        master_arm.rm_movej([0, 0, 0, 0, 0, 0], 20, 0, 0, 1)
        slave_arm.rm_movej([0, 0, 0, 0, 0, 0], 20, 0, 0, 1)
        print("主从臂已复位")

        master_arm.rm_delete_robot_arm()
        slave_arm.rm_delete_robot_arm()
        print("\n停止运行...")

        length = len(save_images)
        for i in range(length):
            image_path = os.path.join(folder_path, f"image_{i}.png")
            cv2.imwrite(image_path, save_images[i])
        print(f"保存图像到: {folder_path}")
        print(f"the number of stored images: {length}")
        # 停止流
        pipeline.stop()


if __name__ == "__main__":
    main_loop()