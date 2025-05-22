import time
import threading
import os
import numpy as np
from Robotic_Arm import *
from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs
import cv2
import datetime

master_trajectory = []
master_time = []
slave_trajectory = []
slave_time = []
save_images = []
image_time = []
last_sent_joint = None
MIN_MOVEMENT_THRESHOLD = 0.05  # 最小运动阈值(弧度)
zhouqi = 10
lock = threading.Lock()
master_queue = None
slave_queue = None

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  # RGB流
profile = pipeline.start(config)
frames = pipeline.wait_for_frames()

def master_state_func(data1):
    global master_trajectory, master_time, lock
    print('主臂也有数据')
    a1 = data1.joint_status.to_dict()
    # b = time.time()
    b1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    master_trajectory.append(a1.copy())
    master_time.append(b1)
    
def slave_state_func(data):
    global slave_trajectory, slave_time, save_images, image_time, pipeline, lock
    print('有数据')
    a = data.joint_status.to_dict()
    frames = pipeline.wait_for_frames(timeout_ms=1000)
    if not frames:
        print("无法获取颜色帧")
    else:
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # b = time.time()
        b = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        slave_trajectory.append(a.copy())
        slave_time.append(b)
        save_images.append(color_image.copy())
        image_time.append(b)


def master_collect(stop_event):
    
    global zhouqi, master_queue

    try:
        master_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle_master = master_arm.rm_create_robot_arm("192.168.110.119", 8080)
    except Exception as e:
        print(f"创建主臂失败: {e}")
        return
    print("创建主臂成功")

    custom = rm_udp_custom_config_t()
    custom.joint_speed = 1
    custom.lift_state = 0
    custom.expand_state = 0
    custom.arm_current_status = 1
    config = rm_realtime_push_config_t(zhouqi, True, 8090, 0, "192.168.110.55", custom)
    print(master_arm.rm_set_realtime_push(config))
    
    arm_state_callback_master = rm_realtime_arm_state_callback_ptr(master_state_func)
    master_arm.rm_realtime_arm_state_call_back(arm_state_callback_master)
    print('start master cycle')

    while not stop_event.is_set():
        start_time = time.time()
        arm_status = master_arm.rm_get_current_arm_state()
        if arm_status[0] != 0:
            print("主臂未处于就绪状态，尝试复位...")
            master_arm.rm_movej([0, 0, 0, 0, 0, 0], 20, 0, 0, 1)
            continue
        else:
            master_queue = arm_status[1]['joint'].copy()
        
        # 计算已用时间
        elapsed_time = time.time() - start_time
        # 动态调整等待时间，确保每次循环大约0.1秒
        sleep_time = max(0, 0.005 - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        
    
    if stop_event.is_set():
        config_end = rm_realtime_push_config_t(zhouqi, False, 8090, 0, "192.168.110.55", custom)
        master_arm.rm_set_realtime_push(config_end)
        print('master arm reset', master_arm.rm_movej([0, 0, 0, 0, 0, 0], 20, 0, 0, 1))
        master_arm.rm_delete_robot_arm()
        print("停止主臂数据采集")


def slave_collect(stop_event):
    global pipeline, zhouqi, MIN_MOVEMENT_THRESHOLD, slave_queue, master_queue

    try:
        slave_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle_slave = slave_arm.rm_create_robot_arm("192.168.110.118", 8080)
    except Exception as e:
        print(f"创建从臂失败: {e}")
        return
    print("创建从臂成功")

    custom = rm_udp_custom_config_t()
    custom.joint_speed = 1
    custom.lift_state = 0
    custom.expand_state = 0
    custom.arm_current_status = 1
    config = rm_realtime_push_config_t(zhouqi, True, 8089, 0, "192.168.110.55", custom)
    print(slave_arm.rm_set_realtime_push(config))

    arm_state_callback_slave = rm_realtime_arm_state_callback_ptr(slave_state_func)
    slave_arm.rm_realtime_arm_state_call_back(arm_state_callback_slave)
    print('start slave cycle')

    while not stop_event.is_set():
        start_time = time.time()
        salve_status = slave_arm.rm_get_current_arm_state()
        if salve_status[0] != 0:
            print("从臂未处于就绪状态，尝试复位...")
            slave_arm.rm_movej([0, 0, 0, 0, 0, 0], 20, 0, 0, 1)
            continue
        slave_queue = salve_status[1]['joint'].copy()
        if master_queue is not None and np.linalg.norm(np.array(master_queue) - np.array(slave_queue)) > MIN_MOVEMENT_THRESHOLD:
            result = slave_arm.rm_movej_follow(master_queue)
            print(f"从臂移动到主臂位置: {master_queue}")
            
            if result != 0:
                print(f"移动指令失败，错误码: {result}，尝试复位...")

        elapsed_time = time.time() - start_time
        # 动态调整等待时间，确保每次循环大约0.1秒
        sleep_time = max(0, 0.005 - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    if stop_event.is_set():
        config_end = rm_realtime_push_config_t(zhouqi, False, 8089, 0, "192.168.110.55", custom)
        slave_arm.rm_set_realtime_push(config_end)
        print('salve arm reset', slave_arm.rm_movej([0, 0, 0, 0, 0, 0], 20, 0, 0, 1))
        slave_arm.rm_delete_robot_arm()
        print("停止从臂数据采集")              

if __name__ == "__main__":
    stop_event = threading.Event()
    try:
        # 把实例传递给线程
        master_collect_thread = threading.Thread(target=master_collect, args=(stop_event, ))
        slave_collect_thread = threading.Thread(target=slave_collect, args=(stop_event, ))
        master_collect_thread.start()
        slave_collect_thread.start()
        # master_slave_thread.start()

        while True:
            # 这里可以添加其他逻辑，比如检查线程状态等
            time.sleep(1)

    except KeyboardInterrupt:
        print('检测到键盘中断，停止数据采集和控制...')
    finally:
        stop_event.set()
        master_collect_thread.join()
        slave_collect_thread.join()
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

        print('data length:', len(master_trajectory), len(slave_trajectory), len(save_images))
        print('total time:', '\n', master_time[-1], master_time[0], '\n', slave_time[-1], slave_time[0], '\n', image_time[-1], image_time[0])
        print("程序结束")
