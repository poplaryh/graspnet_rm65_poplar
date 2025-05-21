import time
import threading
import os
import numpy as np
import queue
from Robotic_Arm import *
from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs
import cv2

master_trajectory = []
master_time = []
slave_trajectory = []
slave_time = []
save_images = []
image_time = []
last_sent_joint = None
MIN_MOVEMENT_THRESHOLD = 0.05  # 最小运动阈值(弧度)
zhouqi = 50

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  # RGB流
profile = pipeline.start(config)
frames = pipeline.wait_for_frames()

slave_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle_slave = slave_arm.rm_create_robot_arm("192.168.110.118", 8080)
master_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle_master = master_arm.rm_create_robot_arm("192.168.110.119", 8080)

def master_state_func(data):
    global master_trajectory, master_time
    a = data.joint_status.to_dict()
    b = time.time()
    master_trajectory.append(a.copy())
    master_time.append(b.copy())
    
def slave_state_func(data):
    global slave_trajectory, slave_time, save_images, image_time, pipeline
    a = data.joint_status.to_dict()
    frames = pipeline.wait_for_frames(timeout_ms=1000)
    if not frames:
        print("无法获取颜色帧")
    else:
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        b = time.time()
        slave_trajectory.append(a.copy())
        slave_time.append(b.copy())
        save_images.append(color_image.copy())
        image_time.append(b.copy())

def master_collect():
    global zhouqi, master_arm, handle_master, stop_event

    # master_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # handle_master = master_arm.rm_create_robot_arm("192.168.110.119", 8080)

    custom = rm_udp_custom_config_t()
    custom.joint_speed = 1
    custom.lift_state = 0
    custom.expand_state = 0
    custom.arm_current_status = 1
    config = rm_realtime_push_config_t(zhouqi, True, 8089, 0, "192.168.110.55", custom)
    print(master_arm.rm_set_realtime_push(config))
    
    arm_state_callback_master = rm_realtime_arm_state_callback_ptr(master_state_func)
    master_arm.rm_realtime_arm_state_call_back(arm_state_callback_master)

    while not stop_event.is_set():
        time.sleep(0.1)
    
    if stop_event.is_set():
        config_end = rm_realtime_push_config_t(zhouqi, False, 8089, 0, "192.168.110.55", custom)
        master_arm.rm_set_realtime_push(config_end)
        master_arm.rm_delete_robot_arm()
        print("停止主臂数据采集")


def slave_collect():
    global pipeline, zhouqi, slave_arm, handle_slave, stop_event

    # slave_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # handle_slave = slave_arm.rm_create_robot_arm("192.168.110.118", 8080)

    custom = rm_udp_custom_config_t()
    custom.joint_speed = 1
    custom.lift_state = 0
    custom.expand_state = 0
    custom.arm_current_status = 1
    config = rm_realtime_push_config_t(zhouqi, True, 8089, 0, "192.168.110.55", custom)
    print(slave_arm.rm_set_realtime_push(config))
    # print(arm2.rm_get_realtime_push())

    arm_state_callback_slave = rm_realtime_arm_state_callback_ptr(slave_state_func)
    slave_arm.rm_realtime_arm_state_call_back(arm_state_callback_slave)

    while not stop_event.is_set():
        time.sleep(0.1)
    
    if stop_event.is_set():
        config_end = rm_realtime_push_config_t(zhouqi, False, 8089, 0, "192.168.110.55", custom)
        slave_arm.rm_set_realtime_push(config_end)
        slave_arm.rm_delete_robot_arm()
        # pipeline.stop()
        print("停止从臂数据采集")

def master_slave_control():

    global last_sent_joint, master_arm, slave_arm, MIN_MOVEMENT_THRESHOLD, stop_event
    print("启动主从臂实时控制...")
    
    # 确保从臂处于正常状态
    slave_status = slave_arm.rm_get_current_arm_state()
    if slave_status[0] != 0:
        print("从臂未处于就绪状态，尝试复位...")
        slave_arm.rm_reset_error()

    master_status = master_arm.rm_get_current_arm_state()
    if master_status[0] != 0:
        print("主臂未处于就绪状态，尝试复位...")
        master_arm.rm_reset_error()    

    while not stop_event.is_set():
        # 记录循环开始时间
        start_time = time.time()
        
        # 获取主臂关节位置
        master_status = master_arm.rm_get_current_arm_state()
        master_joints = master_status[1]['joint']
        
        if master_joints:
            
            # 只有当移动距离超过阈值时才发送命令
            if last_sent_joint is None or np.linalg.norm(np.array(master_joints) - np.array(last_sent_joint)) > MIN_MOVEMENT_THRESHOLD:

                # 直接使用主臂关节位置控制从臂
                result = slave_arm.rm_movej_follow(master_joints)
                
                if result != 0:
                    print(f"移动指令失败，错误码: {result}，尝试复位...")
                    slave_arm.rm_reset_error()
                else:
                    last_sent_joint = master_joints.copy()
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

    master_arm.rm_delete_robot_arm()
    slave_arm.rm_delete_robot_arm()
                

if __name__ == "__main__":
    threading.Lock()
    stop_event = threading.Event()
    try:
        master_collect_thread = threading.Thread(target=master_collect, args=(stop_event, ))
        slave_collect_thread = threading.Thread(target=slave_collect, args=(stop_event, ))
        master_slave_thread = threading.Thread(target=master_slave_control, args=(stop_event, ))
        master_collect_thread.start()
        slave_collect_thread.start()
        master_slave_thread.start()

    except KeyboardInterrupt:
        print('检测到键盘中断，停止数据采集和控制...')
    
    finally:
        stop_event.set()
        master_collect_thread.join()
        slave_collect_thread.join()
        master_slave_thread.join()
        pipeline.stop()

        master_pos = 'master_pos.csv'
        slave_pos = 'slave_pos.csv'
        slave_velocity = 'slave_velocity.csv'
        data_path = os.path.join(os.getcwd(), 'data_image', 'a01')
        master_pos_path = os.path.join(data_path, master_pos)
        slave_pos_path = os.path.join(data_path, slave_pos)
        slave_velocity_path = os.path.join(data_path, slave_velocity)
        folder_path = os.path.join(data_path, 'images')
        # 确保目录存在
        os.makedirs(folder_path, exist_ok=True)

        with open(master_pos_path, 'w', encoding='utf-8') as f:
            for ii, data in enumerate(master_trajectory):
                input_data = [master_time[ii]] + data['joint_position']
                f.write(f"{input_data}\n")
        
        with open(slave_pos_path, 'w', encoding='utf-8') as f:
            for i0, data in enumerate(slave_trajectory):
                pos = data['joint_position']
                input_data = [slave_time[i0]] + pos
                f.write(f"{input_data}\n")

        with open(slave_velocity_path, 'w', encoding='utf-8') as f: 
            for i1, data in enumerate(slave_trajectory):
                vel = data['joint_speed']
                input_data = [slave_time[i1]] + vel
                f.write(f"{input_data}\n")

        for t, img in zip(image_time, save_images):
            image_path = os.path.join(folder_path, f"{t}.png")
            cv2.imwrite(image_path, img)


