import roslibpy
import sys
import cv2
import time
from sensor_msgs.msg import Joy, JointState
from helper_functions.gamepad_functions import GamepadFunctions
from scipy.spatial.transform import Rotation as R
# import rclpy
import os
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray
from copy import deepcopy as cp
import numpy as np
import json
from PIL import Image
from roslibpy import Message
from std_msgs.msg import Header
from adapt_hand_driver.hand_regulator import HandRegulator

import socket
import threading

array = [0]*15

# Global variables to store data from both sources
rgb_data = {
    'joint_data': {},
    'confidence': 0.0,
    'connected': False,
    'timestamp': 0.0
}

rgbd_data = {
    'joint_data': {},
    'confidence': 0.0,
    'connected': False,
    'timestamp': 0.0
}

# Global variables for incremental yaw/pitch positioning
previous_gamepad_state = 0  
held_yaw_position = 0.0     # Hold yaw position when button released
held_pitch_position = 0.0   # Hold pitch position when button released
last_controller_yaw = 0.0   
last_controller_pitch = 0.0 

data_lock = threading.Lock()


def quat_to_euler(_quat_df):
    # print(f'Quat: {quat_df}')
    rot = Rotation.from_quat(_quat_df)

    _rot_euler = rot.as_euler('xyz', degrees=True)
    # print(f'Rot euler: {rot_euler}')
    return _rot_euler

def euler_to_quat(euler_df):
    rot = Rotation.from_euler('xyz', euler_df, degrees=True)
    rot_quat = rot.as_quat()
    # Rotation.from_euler('xyz', rot_euler, degrees=True).as_quat()
    return rot_quat

class Bridge_control:

    def __init__(self) -> None:

        self.ros_client = roslibpy.Ros(host='localhost', port=9090)
        self.ros_client.run()

        self.gp = GamepadFunctions()
        self.hand_regulator = HandRegulator()


        # # Publishers -----------------------------------------
         # fingers - 13 deg
        self.hand_servo_mt_publisher = roslibpy.Topic(self.ros_client, '/adapt_1/hand/servo_demand', 'sensor_msgs/JointState') # '/hand/servo_demand'
        self.hand_servo_mt_publisher.advertise()

        # wrist - 2 deg
        self.target_wrist_joint_names = ['Wrist_Pitch', 'Wrist_Yaw']
        self.wrist_joint_publisher = roslibpy.Topic(self.ros_client, '/adapt_1/wrist/joint_demand', 'sensor_msgs/JointState')
        self.wrist_joint_publisher.advertise()


        # Subscribers -----------------------------------------
        self.gamepad_subscriber = roslibpy.Topic(self.ros_client, '/gamepad', 'sensor_msgs/Joy')
        self.gamepad_subscriber.subscribe(self.gamepad_callback)

        # subscribe /hand_wrist/servo_positions -  read hand 13 dof + wrist 2 dof
        self.hand_wrist_servo_subscriber = roslibpy.Topic(self.ros_client, '/adapt_1/current_servo_data', 'sensor_msgs/JointState')
        self.hand_wrist_servo_subscriber.subscribe(self.hand_wrist_servo_callback)
        


        self.target_hand_servo_names = ['Thumb_CMC2', 'Thumb_CMC1', 'Index_MCP', 'Middle_MCP', 'Ring_MCP', 'Pinky_MCP', 'Pinky_PIP_DIP', 'Ring_PIP_DIP', 'Middle_PIP_DIP', 'Index_PIP_DIP', 'Thumb_MCP', 'Thumb_IP', 'Finger_MCP_Spread']
        self.target_wrist_servo_names = ['Wrist_Thumb_Side', 'Wrist_Pinky_Side']

        self.gamepad_L1 = 0




    def publish_hand_servo_mt(self, mt_demands, init=False):
        # convert to numpy.float64
        mt_demands = np.array(mt_demands, dtype=np.float64)
        # Create a message  
        message = roslibpy.Message({
            "header": {"stamp": { "sec": 0,"nsec": 0 },"frame_id": ""},
            "name":list(self.target_hand_servo_names) ,
            "position":list(mt_demands) # must be list and float64
            })

        self.hand_servo_mt_publisher.publish(message)


        # print(f'Published hand mt demands: {mt_demands}')
        # print(f'Published hand mt demands: {len(mt_demands)}')
        # print("datatype: ", type(list(mt_demands)))
        # print("datatype list(mt_demands)[0]: ", type(list(mt_demands)[0]))
        
    def publish_wrist_joint_mt(self, mt_wrist_demands, init=False):
       # convert to numpy.float64
        mt_wrist_demands = np.array(mt_wrist_demands, dtype=np.float64)
        # Create a message
        message = roslibpy.Message({
            "header": {"stamp": { "sec": 0,"nsec": 0 },"frame_id": ""},
            "name": list(self.target_wrist_joint_names),
            "position": list(mt_wrist_demands) # must be list and float64
            })
        self.wrist_joint_publisher.publish(message)


    def gamepad_callback(self, gamepad_raw_msg:Joy):
        self.gamepad_raw_msg = gamepad_raw_msg
        # print(f'Gamepad: {gamepad_raw_msg}')
        self.gamepad_L1 = gamepad_raw_msg["buttons"][10]
        self.gamepad_L2 = gamepad_raw_msg["buttons"][11]
        self.gamepad_left = gamepad_raw_msg["buttons"][7]
        self.gamepad_R2 = gamepad_raw_msg["buttons"][9]
    
    
    def hand_wrist_servo_callback(self, msg):
        # self.hand_servo_pos = msg["data"][:13]
        # self.wrist_servo_positions = msg["data"][13:]

        self.hand_servo_pos = [0.0] * len(self.target_hand_servo_names) 
        self.wrist_servo_pos = [0.0] * len(self.target_wrist_servo_names)
        # for name, pos in zip(msg["name"], msg["position"]):
        #     for target_name in self.target_hand_servo_names:
        #         if name == target_name:
        #             idx = self.target_hand_servo_names.index(name)
        #             self.hand_servo_pos[idx] = pos

        for idx, target_name in enumerate(self.target_hand_servo_names):
            self.hand_servo_pos[idx] = msg["position"][msg["name"].index(target_name)]
        
        for idx, target_name in enumerate(self.target_wrist_servo_names):
            self.wrist_servo_pos[idx] = msg["position"][msg["name"].index(target_name)]

        pitch, yaw  = self.hand_regulator.wrist_servo_pos_to_pitch_yaw(self.wrist_servo_pos[0], self.wrist_servo_pos[1])
        self.wrist_joint_pos = [pitch, yaw]



    def hand_back_init(self,):
        if self.gp.button_data["L1"] != 1:
        ####################
        # if True:
            if self.gamepad_R2 == 1:
                if self.gp_R2_pressed == False:
                    self.gp_R2_pressed = True
                    self.R2_t = time.time()
                else:
                    if time.time() - self.R2_t > 1 :
                        # raw_demand = cp(self.current_servo_position)
                        # print("self.raw[:6]", self.raw[:6])
                        # self.backinit_step += 1
                        
                        # print("Resetting arm to initial pose  percentage ", self.backinit_step/self.backinit_steps)
                        # print("Resetting arm to initial pose  percentage ", self.backinit_step)
                        self.publish_hand_servo_mt([0]*13)
                        self.publish_wrist_joint_mt([0, 0])
            else:
                self.gp_R2_pressed = False
                self.backinit_step = 0
        ######################

def receive_rgb_joint_data(host='localhost', port=8890):
    """Receive joint data from rgb_retargetting.py"""
    global rgb_data, data_lock
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"RGB joint data receiver listening on {host}:{port}")
        print("Waiting for connection from rgb_retargetting.py...")
        
        while True:
            try:
                client_socket, addr = server_socket.accept()
                print(f"RGB joint data client connected from {addr}")
                
                with data_lock:
                    rgb_data['connected'] = True
                
                buffer = ""
                while True:
                    data = client_socket.recv(4096).decode()
                    if not data:
                        print("RGB connection closed by client")
                        break
                        
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            try:
                                joint_data = json.loads(line)
                                
                                with data_lock:
                                    rgb_data['joint_data'] = joint_data
                                    rgb_data['confidence'] = joint_data.get('hand_confidence', 0.0)
                                    rgb_data['timestamp'] = joint_data.get('timestamp', time.time())

                            except json.JSONDecodeError as e:
                                print(f"RGB JSON decode error: {e}")
                                
            except Exception as e:
                print(f"RGB joint data receiver error: {e}")
            finally:
                try:
                    client_socket.close()
                    with data_lock:
                        rgb_data['connected'] = False
                except:
                    pass
                    
    except Exception as e:
        print(f"RGB server socket error: {e}")
    finally:
        server_socket.close()


def receive_rgbd_joint_data(host='localhost', port=8893):
    """Receive joint data from rgbd_retargetting.py"""
    global rgbd_data, data_lock
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"RGBD joint data receiver listening on {host}:{port}")
        print("Waiting for connection from rgbd_retargetting.py...")
        
        while True:
            try:
                client_socket, addr = server_socket.accept()
                print(f"RGBD joint data client connected from {addr}")
                
                with data_lock:
                    rgbd_data['connected'] = True
                
                buffer = ""
                while True:
                    data = client_socket.recv(4096).decode()
                    if not data:
                        print("RGBD connection closed by client")
                        break
                        
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            try:
                                joint_data = json.loads(line)
                                
                                with data_lock:
                                    rgbd_data['joint_data'] = joint_data
                                    rgbd_data['confidence'] = joint_data.get('confidence', 0.0)  
                                    rgbd_data['timestamp'] = joint_data.get('timestamp', time.time())

                            except json.JSONDecodeError as e:
                                print(f"RGBD JSON decode error: {e}")
                                
            except Exception as e:
                print(f"RGBD joint data receiver error: {e}")
            finally:
                try:
                    client_socket.close()
                    with data_lock:
                        rgbd_data['connected'] = False
                except:
                    pass
                    
    except Exception as e:
        print(f"RGBD server socket error: {e}")
    finally:
        server_socket.close()


def get_best_joint_data():
    """Get joint data from the source with highest confidence"""
    global rgb_data, rgbd_data, data_lock, array
    global previous_gamepad_state, held_yaw_position, held_pitch_position
    global last_controller_yaw, last_controller_pitch
    
    with data_lock:
        # Check which source has higher confidence
        rgb_conf = rgb_data['confidence'] if rgb_data['connected'] else 0.0
        rgbd_conf = rgbd_data['confidence'] if rgbd_data['connected'] else 0.0
        
        # Use data from source with higher confidence for hand control
        if rgb_conf > rgbd_conf and rgb_data['joint_data']:
            selected_data = rgb_data['joint_data']
            source = "RGB"
            confidence = rgb_conf
        elif rgbd_conf > 0.0 and rgbd_data['joint_data']:
            selected_data = rgbd_data['joint_data']
            source = "RGBD"
            confidence = rgbd_conf
        else:
            # No valid data available
            return False, "None", 0.0, [0.0, 0.0, 0.0, 0.0]
        
        # Always extract arm data (world coordinates + roll) from RGBD source if available
        arm_data = [0.0, 0.0, 0.0, 0.0]  # Default values
        if rgbd_data['connected'] and rgbd_data['joint_data']:
            rgbd_joint_data = rgbd_data['joint_data']
            arm_data[0] = rgbd_joint_data.get("wrist_world_x", 0.0) or 0.0  # Handle None values
            arm_data[1] = rgbd_joint_data.get("wrist_world_y", 0.0) or 0.0
            arm_data[2] = rgbd_joint_data.get("wrist_world_z", 0.0) or 0.0
            arm_data[3] = rgbd_joint_data.get("wrist_roll", 0.0) or 0.0
        
        # Get raw yaw and pitch values from controller
        controller_yaw = selected_data.get("Wrist_Yaw_mapped", 0.0)
        controller_pitch = selected_data.get("Wrist_Pitch_mapped", 0.0)
        
        # Update the array with the selected data (all joints except yaw/pitch)
        array[0] = selected_data.get("Thumb_CMC2_mapped", 0.0) 
        array[1] = selected_data.get("Thumb_CMC1_mapped", 0.0)  
        array[2] = selected_data.get("Index_MCP_mapped", 0.0)
        array[3] = selected_data.get("Middle_MCP_MCP_mapped", 0.0)
        array[4] = selected_data.get("Ring_MCP_MCP_mapped", 0.0)
        array[5] = selected_data.get("Pinky_MCP_mapped", 0.0)
        array[6] = selected_data.get("Pinky_PIP_mapped", 0.0)
        array[7] = selected_data.get("Ring_MCP_PIP_mapped", 0.0)
        array[8] = selected_data.get("Middle_MCP_PIP_mapped", 0.0)
        array[9] = selected_data.get("Index_PIP_mapped", 0.0)
        array[10] = selected_data.get("Thumb_MCP_mapped", 0.0)  
        array[11] = selected_data.get("Thumb_IP_mapped", 0.0)  
        array[12] = selected_data.get("Finger_Spread_mapped", 0.0)
        
        # Handle incremental yaw/pitch positioning based on gamepad button state
        current_gamepad_state = bridge.gamepad_L1 if 'bridge' in globals() else 0
        
        if current_gamepad_state == 1:
            if previous_gamepad_state == 0:  # Button was just pressed (transition from 0 to 1)
                # Remember the baseline controller values when button is first pressed
                last_controller_yaw = controller_yaw
                last_controller_pitch = controller_pitch
                print(f"[INCREMENTAL] Button pressed - Starting from held position: Yaw: {held_yaw_position:.2f}, Pitch: {held_pitch_position:.2f}")
            
            # While button is pressed: continuous control = held_position + (current_controller - baseline_controller)
            current_yaw_delta = controller_yaw - last_controller_yaw
            current_pitch_delta = controller_pitch - last_controller_pitch
            
            array[13] = held_pitch_position + current_pitch_delta  
            array[14] = held_yaw_position + current_yaw_delta      
            
        else:  
            if previous_gamepad_state == 1:  
                current_yaw_delta = controller_yaw - last_controller_yaw
                current_pitch_delta = controller_pitch - last_controller_pitch
                held_yaw_position = held_yaw_position + current_yaw_delta
                held_pitch_position = held_pitch_position + current_pitch_delta
                print(f"[INCREMENTAL] Button released - Holding position: Yaw: {held_yaw_position:.2f}, Pitch: {held_pitch_position:.2f}")
            
            # Keep the last held positions
            array[13] = held_pitch_position  
            array[14] = held_yaw_position  
        
        # Update tracking variables
        previous_gamepad_state = current_gamepad_state
                
        return True, source, confidence, arm_data
    

# ================================  MAIN LOOP ================================
if __name__ == "__main__":
    """ bridge = Bridge_control()
     """
    time_start = time.time()    

    rgb_receiver_thread = threading.Thread(target=receive_rgb_joint_data, args=('localhost', 8890), daemon=True)
    rgbd_receiver_thread = threading.Thread(target=receive_rgbd_joint_data, args=('localhost', 8893), daemon=True)

    rgb_receiver_thread.start()
    rgbd_receiver_thread.start()

    while time.time() - time_start < 2000:

        grasp_mtDemands = [0.0] * 15      
        arm_mtDemands = [0.0] * 6

        data_available, source, confidence, arm_data = get_best_joint_data()

        for i in range(15):
            grasp_mtDemands[i] = array[i]
        #print(f'Published hand demands:{grasp_mtDemands}')

        # Use arm data from RGBD (x,y,z,roll, yaw, pitch)
        for i in range(4):
            arm_mtDemands[i] = arm_data[i]
        arm_mtDemands[4] = array[14] 
        arm_mtDemands[5] = array[13] 
        

        hand_demand = grasp_mtDemands[0:13]
        wrist_demand = grasp_mtDemands[13:15]
        print(f'Published hand demands: {hand_demand}, wrist demands: {wrist_demand}')
        #print(f'x,y,z,roll: {arm_mtDemands}')
        #print("bridge.gp.button_data[L1]  ", bridge.gamepad_L1)


        if bridge.gamepad_L1 == 1:
            bridge.publish_hand_servo_mt(hand_demand)
            bridge.publish_wrist_joint_mt(wrist_demand)
        #print(f'yaw:{array[14]:.2f}, pitch: {array[13]:.2f} (held_yaw:{held_yaw_position:.2f}, held_pitch:{held_pitch_position:.2f}), yaw_control:{grasp_mtDemands[14]:.2f}, pitch_control: {grasp_mtDemands[13]:.2f}')
        #print(f'x : {arm_mtDemands[0]},y : {arm_mtDemands[1]},z : {arm_mtDemands[2]}, roll: {arm_mtDemands[3]}')
        time.sleep(0.033)
