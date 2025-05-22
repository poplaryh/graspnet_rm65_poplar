import time
import os
import numpy as np
import datetime
from Robotic_Arm import *
from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs
import cv2
from tqdm import tqdm

# master_trajectory = []
# master_time = []
slave_trajectory = []
slave_time = []
save_images1 = []
save_images2 = []
image_time = []

# def master_state_func(data):
#     global master_trajectory, master_time
#     a = data.joint_status.to_dict()
#     b = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
#     master_trajectory.append(a.copy())
#     master_time.append(b)
    
def slave_state_func(data):
    global slave_trajectory, slave_time, save_images, image_time, pipeline1, pipeline2
    # print('从臂状态回调')
    a = data.joint_status.to_dict()
    frames1 = pipeline1.wait_for_frames(timeout_ms=1000)
    frames2 = pipeline2.wait_for_frames(timeout_ms=1000)
    if not frames1 or not frames2:
        print("无法获取颜色帧")
    else:
        color_frame1 = frames1.get_color_frame()
        color_frame2 = frames2.get_color_frame()
        color_image1 = np.asanyarray(color_frame1.get_data())
        color_image2 = np.asanyarray(color_frame2.get_data())
        b = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        slave_trajectory.append(a.copy())
        slave_time.append(b)
        save_images1.append(color_image1.copy())
        save_images2.append(color_image2.copy())
        image_time.append(b)
               

if __name__ == "__main__":
    last_sent_joint = None
    MIN_MOVEMENT_THRESHOLD = 0.05  # 最小运动阈值(弧度)
    zhouqi = 6
    master_trajectory = []
    master_time = []
    # 确定图像的输入分辨率与帧率
    resolution_width = 640  # pixels
    resolution_height = 480  # pixels
    frame_rate = 60  # fps

    # 注册数据流，并对其图像
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
    # check相机是不是进来了
    connect_device = []
    for d in rs.context().devices:
        print('Found device: ',
              d.get_info(rs.camera_info.name), ' ',
              d.get_info(rs.camera_info.serial_number))
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))

    if len(connect_device) < 2:
        print('Registrition needs two camera connected.But got one.')
        exit()

    # 确认相机并获取相机的内部参数
    pipeline1 = rs.pipeline()
    rs_config.enable_device(connect_device[0])
    pipeline1.start(rs_config)

    pipeline2 = rs.pipeline()
    rs_config.enable_device(connect_device[1])
    pipeline2.start(rs_config)

    frames1 = pipeline1.wait_for_frames()
    frames2 = pipeline2.wait_for_frames()

    # 只在主线程创建
    master_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    slave_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle_master = master_arm.rm_create_robot_arm("192.168.110.119", 8080)
    handle_slave = slave_arm.rm_create_robot_arm("192.168.110.118", 8080)
    print('主臂创建成功', handle_master.id, handle_slave.id)

    custom_slave = rm_udp_custom_config_t()
    custom_slave.joint_speed = 1
    custom_slave.lift_state = 0
    custom_slave.expand_state = 0
    custom_slave.arm_current_status = 1

    # config_master = rm_realtime_push_config_t(zhouqi, True, 8089, 0, "192.168.110.55", custom)
    # print('启动主臂UDP上报', master_arm.rm_set_realtime_push(config_master))
    config_slave = rm_realtime_push_config_t(zhouqi, True, 8089, 0, "192.168.110.55", custom_slave)
    print('启动从臂UDP上报', slave_arm.rm_set_realtime_push(config_slave))

    try:
        slave_status = slave_arm.rm_get_current_arm_state()
        if slave_status[0] != 0:
            print("从臂未处于就绪状态，尝试复位...")

        master_status = master_arm.rm_get_current_arm_state()
        if master_status[0] != 0:
            print("主臂未处于就绪状态，尝试复位...")
        
        arm_state_callback_slave = rm_realtime_arm_state_callback_ptr(slave_state_func)
        slave_arm.rm_realtime_arm_state_call_back(arm_state_callback_slave)
        
        # arm_state_callback_master = rm_realtime_arm_state_callback_ptr(master_state_func)
        # master_arm.rm_realtime_arm_state_call_back(arm_state_callback_master)

        while True:
            # 记录循环开始时间
            start_time = time.time()
            
            # 获取主臂关节位置
            master_status = master_arm.rm_get_current_arm_state()
            master_joints = master_status[1]['joint']
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            
            if master_joints:
                
                # 只有当移动距离超过阈值时才发送命令
                if last_sent_joint is None or np.linalg.norm(np.array(master_joints) - np.array(last_sent_joint)) > MIN_MOVEMENT_THRESHOLD:

                    # 直接使用主臂关节位置控制从臂
                    result = slave_arm.rm_movej_follow(master_joints)
                    # print(f"从臂移动到主臂位置: {master_joints}")
                    
                    if result != 0:
                        print(f"移动指令失败，错误码: {result}，尝试复位...")
                    else:
                        last_sent_joint = master_joints.copy()
                        
                        master_trajectory.append(master_joints.copy())
                        master_time.append(time_stamp)
                # else:
                    # print("变化量小于阈值，跳过")
            else:
                print('无法获取主臂数据') 
            
            # 计算已用时间
            elapsed_time = time.time() - start_time
            # 动态调整等待时间，确保每次循环大约0.1秒
            sleep_time = max(0, 0.01 - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print('检测到键盘中断，停止数据采集和控制...')

    finally:
        # config_end_master = rm_realtime_push_config_t(zhouqi, False, 8089, 0, "192.168.110.55", custom)
        config_end_slave = rm_realtime_push_config_t(zhouqi, False, 8089, 0, "192.168.110.55", custom_slave)
        # print("停止主臂数据采集", master_arm.rm_set_realtime_push(config_end_master))
        master_arm.rm_movej([0, 0, 0, 0, 0, 0], 40, 0, 0, 1)
        slave_arm.rm_movej([0, 0, 0, 0, 0, 0], 40, 0, 0, 1)
        print("停止从臂数据采集", slave_arm.rm_set_realtime_push(config_end_slave))
        print('从臂删除成功', slave_arm.rm_delete_robot_arm())
        print('主臂删除成功', master_arm.rm_delete_robot_arm())

        pipeline1.stop()
        pipeline2.stop()

        master_pos = 'master_pos.csv'
        slave_pos = 'slave_pos.csv'
        slave_velocity = 'slave_velocity.csv'
        data_path = os.path.join(os.getcwd(), 'data_image', 'a02')
        master_pos_path = os.path.join(data_path, master_pos)
        slave_pos_path = os.path.join(data_path, slave_pos)
        slave_velocity_path = os.path.join(data_path, slave_velocity)
        folder_path1 = os.path.join(data_path, 'images', 'cam1')
        folder_path2 = os.path.join(data_path, 'images', 'cam2')
        # 确保目录存在
        os.makedirs(folder_path1, exist_ok=True)
        os.makedirs(folder_path2, exist_ok=True)

        print('data length:', len(master_trajectory), len(slave_trajectory), len(save_images1), len(save_images2))
        print('time length:', len(master_time), len(slave_time), len(image_time))

        # 保存主臂位置
        with open(master_pos_path, 'w', encoding='utf-8') as f:
            for ii, data in tqdm(enumerate(master_trajectory), total=len(master_trajectory), desc="Saving master pos"):
                input_data = [master_time[ii]] + data
                f.write(f"{input_data}\n")

        # 保存从臂位置
        with open(slave_pos_path, 'w', encoding='utf-8') as f:
            for i0, data in tqdm(enumerate(slave_trajectory), total=len(slave_trajectory), desc="Saving slave pos"):
                pos = data['joint_position']
                input_data = [slave_time[i0].replace(':', '-').replace(' ', '_')] + pos
                f.write(f"{input_data}\n")

        # 保存从臂速度
        with open(slave_velocity_path, 'w', encoding='utf-8') as f: 
            for i1, data in tqdm(enumerate(slave_trajectory), total=len(slave_trajectory), desc="Saving slave vel"):
                vel = data['joint_speed']
                input_data = [slave_time[i1].replace(':', '-').replace(' ', '_')] + vel
                f.write(f"{input_data}\n")

        # 保存相机1图片
        for i2, img in tqdm(enumerate(save_images1), total=len(save_images1), desc="Saving cam1 images"):
            tmp_time = image_time[i2].replace(':', '-').replace(' ', '_')
            image_path = os.path.join(folder_path1, f"{tmp_time}.png")
            cv2.imwrite(image_path, img)

        # 保存相机2图片
        for i3, img3 in tqdm(enumerate(save_images2), total=len(save_images2), desc="Saving cam2 images"):
            tmp_time3 = image_time[i3].replace(':', '-').replace(' ', '_')
            image_path3 = os.path.join(folder_path2, f"{tmp_time3}.png")
            cv2.imwrite(image_path3, img3)

        # print('total time:', master_time[-1] - master_time[0], slave_time[-1] - slave_time[0], image_time[-1] - image_time[0])
        print("程序结束")
        RoboticArm.rm_destory()
