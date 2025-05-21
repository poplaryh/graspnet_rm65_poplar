import socket
import json
import sys
import time
from robotic_arm import *

#设置主动上报
#{"command":"set_realtime_push","cycle":2,"enable":true,"port":8089,"ip":"192.168.1.111"}

# IP and port configuration 主臂
ip = '192.168.1.19'
port_no = 8080
# Create a socket and connect to the server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, port_no))
print("机械臂第一次连接", ip)

ip = '192.168.1.18' # 从动机械臂
client_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_1.connect((ip, port_no))
print("机械臂第一次连接", ip)
commod = '{"command":"start_drag_teach","trajectory_record":0}\r\n'
client_1.sendall(commod.encode('utf-8'))
time.sleep(3)

# 创建UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('192.168.1.111', 8089))
print("Listening for incoming messages on port 8089...")
# 设置一个合适的超时值以避免无限期等待
sock.settimeout(10)

while 1 :
    # 尝试接收数据
    received_data, _ = sock.recvfrom(1024)
    print("Received data:", received_data.decode('utf-8'))

    # 将接收到的数据解析为JSON格式的字典
    data_dict = json.loads(received_data.decode('utf-8'))

    # 从解析后的字典中提取joint_position字段的值
    joint = data_dict["joint_status"]["joint_position"]
    print("Joint positions:", joint)
    message = {
        "command": "movej_canfd",
        "joint": joint,
        "follow": False,
        "expand": 0
    }
    # 将字典转换为JSON格式的字符串
    json_message = json.dumps(message)
    client.sendall(json_message.encode('utf-8'))

