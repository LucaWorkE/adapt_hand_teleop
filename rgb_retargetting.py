"""

This script sends finger angle and 
wrist orientation data to Isaac Sim via TCP on port 8888.
Depth is inferred from mediapipe

Dependencies:
- mediapipe
- opencv-python
- numpy

"""

import multiprocessing
import time
import socket
import struct
import pickle
import threading
import json
import argparse
from pathlib import Path
from queue import Empty
from typing import Optional
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger


def calculate_single_finger_angles(joint_pos, finger_landmarks):
    """
    Calculate joint angles for a single finger from MediaPipe landmarks.
    Now calculates MCP, PIP, and DIP angles plus spread.
    
    Args:
        joint_pos: MediaPipe joint positions array
        finger_landmarks: List of [mcp_idx, pip_idx, dip_idx, tip_idx] landmark indices
        
    Returns:
        dict: Joint angles in radians for the single finger
    """
    if joint_pos is None:
        return None
        
    try:
        wrist_pos = joint_pos[0]  # Wrist position (x,y,z)
        mcp_pos = joint_pos[finger_landmarks[0]]  # MCP position
        pip_pos = joint_pos[finger_landmarks[1]]  # PIP position
        dip_pos = joint_pos[finger_landmarks[2]]  # DIP position
        tip_pos = joint_pos[finger_landmarks[3]]  # Tip position
        
        # Calculate vectors for all segments
        mcp_to_pip_vec = pip_pos - mcp_pos
        mcp_to_pip_vec = mcp_to_pip_vec / np.linalg.norm(mcp_to_pip_vec)
        
        pip_to_dip_vec = dip_pos - pip_pos
        pip_to_dip_vec = pip_to_dip_vec / np.linalg.norm(pip_to_dip_vec)
        
        dip_to_tip_vec = tip_pos - dip_pos
        dip_to_tip_vec = dip_to_tip_vec / np.linalg.norm(dip_to_tip_vec)
        
        wrist_to_mcp_vec = mcp_pos - wrist_pos
        wrist_to_mcp_vec = wrist_to_mcp_vec / np.linalg.norm(wrist_to_mcp_vec)
        
        # Calculate joint flexion angles (MCP, PIP, and DIP)
        mcp_flex_angle = np.pi - np.arccos(np.clip(np.dot(wrist_to_mcp_vec, mcp_to_pip_vec), -1, 1))
        pip_flex_angle = np.pi - np.arccos(np.clip(np.dot(mcp_to_pip_vec, pip_to_dip_vec), -1, 1))
        dip_flex_angle = np.pi - np.arccos(np.clip(np.dot(pip_to_dip_vec, dip_to_tip_vec), -1, 1))
        
        # Calculate spread angle
        mcp_spread_angle = np.arctan2(mcp_to_pip_vec[1], np.sqrt(mcp_to_pip_vec[0]**2 + mcp_to_pip_vec[2]**2))
        
        return {
            'mcp_spread': mcp_spread_angle,
            'mcp_flex': mcp_flex_angle,
            'pip_flex': pip_flex_angle,
            'dip_flex': dip_flex_angle
        }
        
    except Exception as e:
        return None


def calculate_all_finger_angles(joint_pos):
    """ 
    Returns:
        dict: Joint angles in radians for all fingers
    """
    if joint_pos is None:
        return None
        
    try:
        # Define finger landmark indices from mediapipe idx: [MCP, PIP, DIP, TIP]
        finger_landmarks = {
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        all_finger_angles = {}
        
        # Calculate angles for each finger
        for finger_name, landmarks in finger_landmarks.items():
            finger_angles = calculate_single_finger_angles(joint_pos, landmarks)
            if finger_angles is not None:
                all_finger_angles[finger_name] = finger_angles
        
        # For backward compatibility, return index finger angles at top level
        if 'index' in all_finger_angles:
            result = all_finger_angles['index'].copy()
            result['all_fingers'] = all_finger_angles
            return result
        else:
            return None
        
    except Exception as e:
        return None


def calculate_thumb_joint_angles(joint_pos):
    """ 
    Returns:
        dict: Thumb joint angles in degrees
    """
    if joint_pos is None:
        return None
        
    try:
        # MediaPipe thumb landmark positions
        wrist_pos = joint_pos[0]    # Wrist position (reference)
        cmc_pos = joint_pos[1]      # CMC joint (base of thumb)
        mcp_pos = joint_pos[2]      # MCP joint 
        ip_pos = joint_pos[3]       # IP joint (equivalent to PIP)
        tip_pos = joint_pos[4]      # Thumb tip
        
        # Calculate vectors for each segment
        wrist_to_cmc_vec = cmc_pos - wrist_pos
        cmc_to_mcp_vec = mcp_pos - cmc_pos
        mcp_to_ip_vec = ip_pos - mcp_pos
        ip_to_tip_vec = tip_pos - ip_pos
        
        # Normalize vectors
        wrist_to_cmc_vec = wrist_to_cmc_vec / np.linalg.norm(wrist_to_cmc_vec)
        cmc_to_mcp_vec = cmc_to_mcp_vec / np.linalg.norm(cmc_to_mcp_vec)
        mcp_to_ip_vec = mcp_to_ip_vec / np.linalg.norm(mcp_to_ip_vec)
        ip_to_tip_vec = ip_to_tip_vec / np.linalg.norm(ip_to_tip_vec)
        
        # Calculate joint angles
        cmc1_angle = np.arctan2(wrist_to_cmc_vec[1], np.sqrt(wrist_to_cmc_vec[0]**2 + wrist_to_cmc_vec[2]**2))
        cmc2_angle = np.pi - np.arccos(np.clip(np.dot(wrist_to_cmc_vec, cmc_to_mcp_vec), -1, 1))
        mcp_angle = np.pi - np.arccos(np.clip(np.dot(cmc_to_mcp_vec, mcp_to_ip_vec), -1, 1))
        ip_angle = np.pi - np.arccos(np.clip(np.dot(mcp_to_ip_vec, ip_to_tip_vec), -1, 1))
        
        # Convert to degrees
        return {
            'cmc1_deg': np.degrees(cmc1_angle),
            'cmc2_deg': np.degrees(cmc2_angle),
            'mcp_deg': np.degrees(mcp_angle),
            'ip_deg': np.degrees(ip_angle)
        }
        
    except Exception as e:
        return None


def compute_yaw_pitch(rotation_matrix):

    if rotation_matrix is None:
        return 0.0, 0.0
        
    try:
        # Extract yaw (rotation around Z-axis)
        forward_x = rotation_matrix[0, 0]
        forward_y = rotation_matrix[1, 0]
        yaw = np.arctan2(forward_y, forward_x)
        
        # Extract pitch (rotation around X-axis)
        forward_z = rotation_matrix[2, 0]
        forward_xy_magnitude = np.sqrt(forward_x**2 + forward_y**2)
        pitch = np.arctan2(-forward_z, forward_xy_magnitude)
        
        return yaw, pitch
        
    except Exception as e:
        return 0.0, 0.0


def compute_roll(rotation_matrix):
    
    if rotation_matrix is None:
        return 0.0
        
    try:
        up_vector = rotation_matrix[:, 2]
        right_vector = rotation_matrix[:, 1]
        
        roll = np.arctan2(right_vector[2], up_vector[2])
        
        return roll
        
    except Exception as e:
        return 0.0


class MediaPipeHandTracker:
    """
    MediaPipe-based hand tracker that detects landmarks and transforms them to MANO convention.
    Processes RGB images to extract 3D hand pose, joint positions, and wrist orientation.
    """
    
    # Hand orientation matrices (copied from anyteleop : dex-retargeting)
    OPERATOR2MANO_RIGHT = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ])
    
    OPERATOR2MANO_LEFT = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ])

    def __init__(self, hand_type="Right", min_detection_confidence=0.8, min_tracking_confidence=0.8, selfie=False):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = (
            self.OPERATOR2MANO_RIGHT if hand_type == "Right" else self.OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    def detect(self, rgb_image):
        """
        Detect hand landmarks and return processed data.
        
        Returns:
            tuple: (num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot, confidence)
        """
        results = self.hand_detector.process(rgb_image)
        
        if results.multi_hand_landmarks is None or results.multi_handedness is None:
            return 0, None, None, None, 0.0
            
        # Get confidence from MediaPipe detection
        hand_confidence = results.multi_handedness[0].classification[0].score
            
        # Process first detected hand
        keypoint_3d = results.multi_hand_landmarks[0]
        keypoint_2d = results.multi_hand_landmarks[0]
        
        # Parse 3D keypoints
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        
        # Estimate hand frame
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        
        # Transform to MANO convention
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano
        
        return 1, joint_pos, keypoint_2d, mediapipe_wrist_rot, hand_confidence

    @staticmethod
    def parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        """Parse 3D keypoints from MediaPipe output."""
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame from detected 3D keypoints.
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]  # wrist, index MCP, middle MCP

        # Compute vector from palm to middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)
        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # Orient Z axis correctly
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
            
        frame = np.stack([x, normal, z], axis=1)
        return frame


# Global variable to store latest joint data
latest_joint_data = {
    'wrist_camera_x': None,
    'wrist_camera_y': None,
    'wrist_world_x': None,
    'wrist_world_y': None,
    'wrist_world_z': None,
    'yaw_deg': None,
    'pitch_deg': None,
    'roll_deg': None,
    'index_mcp_spread_deg': None,
    'index_mcp_flex_deg': None,
    'index_pip_flex_deg': None,
    'index_dip_flex_deg': None,
    'middle_mcp_spread_deg': None,
    'middle_mcp_flex_deg': None,
    'middle_pip_flex_deg': None,
    'middle_dip_flex_deg': None,
    'ring_mcp_spread_deg': None,
    'ring_mcp_flex_deg': None,
    'ring_pip_flex_deg': None,
    'ring_dip_flex_deg': None,
    'pinky_mcp_spread_deg': None,
    'pinky_mcp_flex_deg': None,
    'pinky_pip_flex_deg': None,
    'pinky_dip_flex_deg': None,
    'thumb_cmc1_deg': None,
    'thumb_cmc2_deg': None,
    'thumb_mcp_deg': None,
    'thumb_ip_deg': None,
    'depth_frame': None,
    'timestamp': None
}
joint_data_lock = threading.Lock()


def joint_data_server(host='0.0.0.0', port=8888):
    """Server to send hand data to Isaac Sim"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        
        while True:
            try:
                client_socket, addr = server_socket.accept()
                
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                last_send_time = time.time()
                last_heartbeat = time.time()
                
                while True:
                    try:
                        # Check for heartbeat
                        client_socket.settimeout(0.1)
                        try:
                            data = client_socket.recv(1024)
                            if data:
                                try:
                                    msg = json.loads(data.decode().strip())
                                    if msg.get('type') == 'heartbeat':
                                        last_heartbeat = time.time()
                                except:
                                    pass
                        except socket.timeout:
                            pass
                        
                        # Check client alive
                        if time.time() - last_heartbeat > 30:
                            break
                        
                        # Send data at 30 FPS
                        current_time = time.time()
                        if current_time - last_send_time >= 1/30:
                            with joint_data_lock:
                                # Always send data, even when no hand is detected
                                # This ensures confidence values are always up-to-date
                                
                                # Extract depth value at wrist
                                wrist_depth_value = None
                                depth_frame = latest_joint_data.get('depth_frame')
                                wrist_x = latest_joint_data.get('wrist_camera_x')
                                wrist_y = latest_joint_data.get('wrist_camera_y')
                                
                                if depth_frame is not None and wrist_x is not None and wrist_y is not None:
                                    try:
                                        x = int(np.clip(wrist_x, 0, depth_frame.shape[1] - 1))
                                        y = int(np.clip(wrist_y, 0, depth_frame.shape[0] - 1))
                                        wrist_depth_value = float(depth_frame[y, x])
                                    except Exception as e:
                                        pass
                                
                                # Use current timestamp if none available (for when no hand detected)
                                data_timestamp = latest_joint_data.get('timestamp') or current_time
                                
                                data_to_send = {
                                    'yaw_deg': latest_joint_data.get('yaw_deg', 71.0),
                                    'pitch_deg': latest_joint_data.get('pitch_deg', -30.0),
                                    'roll_deg': latest_joint_data.get('roll_deg', 0.0),
                                    'wrist_camera_x': latest_joint_data.get('wrist_camera_x'),
                                    'wrist_camera_y': latest_joint_data.get('wrist_camera_y'),
                                    'wrist_depth_z': wrist_depth_value,
                                    'wrist_world_x': latest_joint_data.get('wrist_world_x'),
                                    'wrist_world_y': latest_joint_data.get('wrist_world_y'),
                                    'wrist_world_z': latest_joint_data.get('wrist_world_z'),
                                    'depth_available': depth_frame is not None,
                                    'index_mcp_spread_deg': latest_joint_data.get('index_mcp_spread_deg'),
                                    'index_mcp_flex_deg': latest_joint_data.get('index_mcp_flex_deg'),
                                    'index_pip_flex_deg': latest_joint_data.get('index_pip_flex_deg'),
                                    'index_dip_flex_deg': latest_joint_data.get('index_dip_flex_deg'),
                                    'middle_mcp_spread_deg': latest_joint_data.get('middle_mcp_spread_deg'),
                                    'middle_mcp_flex_deg': latest_joint_data.get('middle_mcp_flex_deg'),
                                    'middle_pip_flex_deg': latest_joint_data.get('middle_pip_flex_deg'),
                                    'middle_dip_flex_deg': latest_joint_data.get('middle_dip_flex_deg'),
                                    'ring_mcp_spread_deg': latest_joint_data.get('ring_mcp_spread_deg'),
                                    'ring_mcp_flex_deg': latest_joint_data.get('ring_mcp_flex_deg'),
                                    'ring_pip_flex_deg': latest_joint_data.get('ring_pip_flex_deg'),
                                    'ring_dip_flex_deg': latest_joint_data.get('ring_dip_flex_deg'),
                                    'pinky_mcp_spread_deg': latest_joint_data.get('pinky_mcp_spread_deg'),
                                    'pinky_mcp_flex_deg': latest_joint_data.get('pinky_mcp_flex_deg'),
                                    'pinky_pip_flex_deg': latest_joint_data.get('pinky_pip_flex_deg'),
                                    'pinky_dip_flex_deg': latest_joint_data.get('pinky_dip_flex_deg'),
                                    'thumb_cmc1_deg': latest_joint_data.get('thumb_cmc1_deg'),
                                    'thumb_cmc2_deg': latest_joint_data.get('thumb_cmc2_deg'),
                                    'thumb_mcp_deg': latest_joint_data.get('thumb_mcp_deg'),
                                    'thumb_ip_deg': latest_joint_data.get('thumb_ip_deg'),
                                    'timestamp': data_timestamp,
                                    'hand_confidence': latest_joint_data.get('hand_confidence', 0.0)  
                                }
                                
                                # Send as JSON
                                message = json.dumps(data_to_send) + '\n'
                                client_socket.send(message.encode())
                                last_send_time = current_time
                        
                        time.sleep(0.01)
                        
                    except (ConnectionResetError, BrokenPipeError, OSError) as e:
                        break
                    except Exception as e:
                        break
                
            except Exception as e:
                pass
            finally:
                try:
                    client_socket.close()
                except:
                    pass
                    
    except Exception as e:
        pass
    finally:
        server_socket.close()


def start_hand_tracking(queue: multiprocessing.Queue, depth_queue: multiprocessing.Queue, hand_type: str):
    global latest_joint_data, joint_data_lock
    
    detector = MediaPipeHandTracker(hand_type=hand_type, selfie=False)

    # Start data server
    server_thread = threading.Thread(target=joint_data_server, daemon=True)
    server_thread.start()

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            return

        # Get depth frame
        current_depth_frame = None
        try:
            while not depth_queue.empty():
                current_depth_frame = depth_queue.get_nowait()
        except:
            pass

        # Detect hand
        _, joint_pos, keypoint_2d, wrist_rot_matrix, hand_confidence = detector.detect(rgb)

        if joint_pos is None:
            with joint_data_lock:
                latest_joint_data['yaw_deg'] = None
                latest_joint_data['pitch_deg'] = None
                latest_joint_data['roll_deg'] = None
                latest_joint_data['timestamp'] = None
                latest_joint_data['hand_confidence'] = 0.0  # No hand detected
        else:
            # Extract wrist position in camera coordinates
            wrist_camera_x = None
            wrist_camera_y = None
            if keypoint_2d is not None and len(keypoint_2d.landmark) > 0:
                wrist_landmark = keypoint_2d.landmark[0]
                wrist_camera_x = wrist_landmark.x * 640
                wrist_camera_y = wrist_landmark.y * 480
            
            # Extract orientation
            yaw_independent, pitch_independent = compute_yaw_pitch(wrist_rot_matrix)
            roll_v2 = compute_roll(wrist_rot_matrix)
            
            # Convert to degrees and apply calibration
            yaw_deg = np.degrees(yaw_independent) + 71.0
            pitch_deg = np.degrees(pitch_independent) - 30.0
            roll_deg = np.degrees(roll_v2)
            
            # RGB camera doesn't provide world coordinates - always set to None
            wrist_world_x, wrist_world_y, wrist_world_z = None, None, None
            
            # Calculate finger angles
            direct_angles = calculate_all_finger_angles(joint_pos)
            thumb_angles = calculate_thumb_joint_angles(joint_pos)
            
            # Update global data
            with joint_data_lock:
                latest_joint_data['yaw_deg'] = yaw_deg
                latest_joint_data['pitch_deg'] = pitch_deg
                latest_joint_data['roll_deg'] = roll_deg
                latest_joint_data['wrist_camera_x'] = wrist_camera_x
                latest_joint_data['wrist_camera_y'] = wrist_camera_y
                latest_joint_data['wrist_world_x'] = wrist_world_x
                latest_joint_data['wrist_world_y'] = wrist_world_y
                latest_joint_data['wrist_world_z'] = wrist_world_z
                latest_joint_data['depth_frame'] = current_depth_frame
                latest_joint_data['timestamp'] = time.time()
                latest_joint_data['hand_confidence'] = hand_confidence  # Store MediaPipe confidence
                
                # Store finger angles
                if direct_angles is not None:
                    latest_joint_data['index_mcp_spread_deg'] = np.degrees(direct_angles['mcp_spread'])
                    latest_joint_data['index_mcp_flex_deg'] = np.degrees(direct_angles['mcp_flex'])
                    latest_joint_data['index_pip_flex_deg'] = np.degrees(direct_angles['pip_flex'])
                    latest_joint_data['index_dip_flex_deg'] = np.degrees(direct_angles['dip_flex'])
                    
                    if 'all_fingers' in direct_angles:
                        all_fingers = direct_angles['all_fingers']
                        
                        # Middle finger
                        if 'middle' in all_fingers:
                            latest_joint_data['middle_mcp_spread_deg'] = np.degrees(all_fingers['middle']['mcp_spread'])
                            latest_joint_data['middle_mcp_flex_deg'] = np.degrees(all_fingers['middle']['mcp_flex'])
                            latest_joint_data['middle_pip_flex_deg'] = np.degrees(all_fingers['middle']['pip_flex'])
                            latest_joint_data['middle_dip_flex_deg'] = np.degrees(all_fingers['middle']['dip_flex'])
                        
                        # Ring finger
                        if 'ring' in all_fingers:
                            latest_joint_data['ring_mcp_spread_deg'] = np.degrees(all_fingers['ring']['mcp_spread'])
                            latest_joint_data['ring_mcp_flex_deg'] = np.degrees(all_fingers['ring']['mcp_flex'])
                            latest_joint_data['ring_pip_flex_deg'] = np.degrees(all_fingers['ring']['pip_flex'])
                            latest_joint_data['ring_dip_flex_deg'] = np.degrees(all_fingers['ring']['dip_flex'])
                        
                        # Pinky finger
                        if 'pinky' in all_fingers:
                            latest_joint_data['pinky_mcp_spread_deg'] = np.degrees(all_fingers['pinky']['mcp_spread'])
                            latest_joint_data['pinky_mcp_flex_deg'] = np.degrees(all_fingers['pinky']['mcp_flex'])
                            latest_joint_data['pinky_pip_flex_deg'] = np.degrees(all_fingers['pinky']['pip_flex'])
                            latest_joint_data['pinky_dip_flex_deg'] = np.degrees(all_fingers['pinky']['dip_flex'])
                
                # Store thumb angles
                if thumb_angles is not None:
                    latest_joint_data['thumb_cmc1_deg'] = thumb_angles['cmc1_deg']
                    latest_joint_data['thumb_cmc2_deg'] = thumb_angles['cmc2_deg']
                    latest_joint_data['thumb_mcp_deg'] = thumb_angles['mcp_deg']
                    latest_joint_data['thumb_ip_deg'] = thumb_angles['ip_deg']


def produce_frame_from_network(queue, depth_queue, host='192.168.1.152', port=9999):
    """Receive frames from network camera"""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        logger.info(f"Connected to camera server at {host}:{port}")
        
        data = b""
        payload_size = struct.calcsize(">L")

        while True:
            # Retrieve message size
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet
            if len(data) < payload_size:
                break
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            # Retrieve all data
            while len(data) < msg_size:
                data += client_socket.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Deserialize data package
            try:
                data_package = pickle.loads(frame_data)
                if isinstance(data_package, dict):
                    color_frame = data_package.get('color_frame')
                    depth_frame = data_package.get('depth_frame')
                    
                    # Store depth frame globally
                    global latest_joint_data, joint_data_lock
                    with joint_data_lock:
                        latest_joint_data['depth_frame'] = depth_frame
                    
                    # Put depth frame in queue
                    if not depth_queue.full():
                        depth_queue.put(depth_frame)
                    
                    frame = color_frame
                else:
                    frame = data_package
                    with joint_data_lock:
                        latest_joint_data['depth_frame'] = None
                    
                    if not depth_queue.full():
                        depth_queue.put(None)
            except Exception as e:
                frame = pickle.loads(frame_data)
                with joint_data_lock:
                    latest_joint_data['depth_frame'] = None
                
                if not depth_queue.full():
                    depth_queue.put(None)

            # Put frame in queue
            if not queue.full():
                queue.put(frame)
                
    except Exception as e:
        logger.error(f"Camera connection error: {e}")
    finally:
        try:
            client_socket.close()
        except:
            pass


def get_joint_data():
    """
    Get all available 20 joint data
    
    Returns:
        dict: All joint angles in degrees with standardized naming
    """
    with joint_data_lock:
        return {
            # Index finger
            'Index_MCP_Spread': latest_joint_data.get('index_mcp_spread_deg'),
            'Index_MCP': latest_joint_data.get('index_mcp_flex_deg'),
            'Index_PIP': latest_joint_data.get('index_pip_flex_deg'),
            'Index_DIP': latest_joint_data.get('index_dip_flex_deg'),
            
            # Middle finger
            'Middle_MCP_Spread': latest_joint_data.get('middle_mcp_spread_deg'),
            'Middle_MCP': latest_joint_data.get('middle_mcp_flex_deg'),
            'Middle_PIP': latest_joint_data.get('middle_pip_flex_deg'),
            'Middle_DIP': latest_joint_data.get('middle_dip_flex_deg'),
            
            # Ring finger
            'Ring_MCP_Spread': latest_joint_data.get('ring_mcp_spread_deg'),
            'Ring_MCP': latest_joint_data.get('ring_mcp_flex_deg'),
            'Ring_PIP': latest_joint_data.get('ring_pip_flex_deg'),
            'Ring_DIP': latest_joint_data.get('ring_dip_flex_deg'),
            
            # Pinky finger
            'Pinky_MCP_Spread': latest_joint_data.get('pinky_mcp_spread_deg'),
            'Pinky_MCP': latest_joint_data.get('pinky_mcp_flex_deg'),
            'Pinky_PIP': latest_joint_data.get('pinky_pip_flex_deg'),
            'Pinky_DIP': latest_joint_data.get('pinky_dip_flex_deg'),
            
            # Thumb
            'Thumb_CMC1': latest_joint_data.get('thumb_cmc1_deg'),
            'Thumb_CMC2': latest_joint_data.get('thumb_cmc2_deg'),
            'Thumb_MCP': latest_joint_data.get('thumb_mcp_deg'),
            'Thumb_IP': latest_joint_data.get('thumb_ip_deg')
        }


def main(hand_type="Right", camera_host="192.168.1.152", camera_port=9999):
    """
    Standalone hand tracker
    
    Args:
        hand_type: "Right" or "Left"
        camera_host: Camera server IP
        camera_port: Camera server port
    """
    print(f"Camera Server: {camera_host}:{camera_port}")
    print(f"Detecting {hand_type} hand")
    print(f"Standalone hand tracker")
    print("-" * 80)

    queue = multiprocessing.Queue(maxsize=50)
    depth_queue = multiprocessing.Queue(maxsize=20)
    
    producer_process = multiprocessing.Process(
        target=produce_frame_from_network, args=(queue, depth_queue, camera_host, camera_port)
    )
    consumer_process = multiprocessing.Process(
        target=start_hand_tracking, args=(queue, depth_queue, hand_type)
    )

    logger.info("Starting standalone hand tracker...")
    logger.info(f"Connecting to camera server at {camera_host}:{camera_port}")
    logger.info("SERVER: Starting TCP server on port 8888 for Isaac Lab")
    
    producer_process.start()
    consumer_process.start()

    try:
        producer_process.join()
        consumer_process.join()
    except KeyboardInterrupt:
        print("\n Interrupted by user")
        logger.info("Terminating processes...")
        producer_process.terminate()
        consumer_process.terminate()
        
        time.sleep(1)
        
        if producer_process.is_alive():
            producer_process.kill()
        if consumer_process.is_alive():
            consumer_process.kill()
    
    time.sleep(1)
    print(" Done")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Hand Tracker for Isaac Sim")
    parser.add_argument("--hand-type", choices=["Right", "Left"], default="Right", help="Hand to track")
    parser.add_argument("--camera-host", default="192.168.1.152", help="Camera server host")
    parser.add_argument("--camera-port", type=int, default=9999, help="Camera server port")
    
    args = parser.parse_args()
    main(args.hand_type, args.camera_host, args.camera_port)
