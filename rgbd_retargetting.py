"""
Sends finger angle and 
wrist orientation data to Isaac Sim via TCP on port 8889.

This version works with camera_server_intel.py which provides both RGB and depth data
plus camera intrinsics from Intel RealSense cameras on port 9998.

Dependencies:
- mediapipe
- opencv-python
- numpy
- pyrealsense2 (for intrinsics structure)
- loguru (optional, can be replaced with standard logging)
"""

import multiprocessing
import time
import socket
import struct
import pickle
import threading
import json
import argparse
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from queue import Empty
from typing import Optional
from loguru import logger
from typing import Optional
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from loguru import logger
import sys

# Ubuntu keyboard handling with pynput
try:
    import pynput
    from pynput import keyboard as pynput_keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("[WARNING] Calibration features will be disabled.")


# Global calibration variables
calibration_data = {
    'yaw_neutral': 0.0,
    'yaw_min': None,
    'yaw_max': None,
    'pitch_neutral': 0.0,
    'pitch_min': None,
    'pitch_max': None,
    'yaw_calibrating': False,
    'pitch_calibrating': False
}
calibration_lock = threading.Lock()

# Global state for pynput keyboard handling
current_keys_pressed = set()
keys_lock = threading.Lock()


def keyboard_calibration_handler():
    global current_keys_pressed, keys_lock

    if not KEYBOARD_AVAILABLE:
        print("[CALIBRATION] pynput library not available - calibration disabled")
        return
    
    print("[CALIBRATION] Press 'y' to calibrate yaw (hold while moving hand)")
    print("[CALIBRATION] Press 'p' to calibrate pitch (hold while moving hand)")
    print("[CALIBRATION] Press 'q' to quit")
    

    # Setup pynput keyboard listener
    def on_press(key):
        try:
            char = key.char
            if char in ['y', 'p', 'q']:
                with keys_lock:
                    current_keys_pressed.add(char)
                    print(f"[DEBUG] Key '{char}' pressed and added to set")
        except AttributeError:
            pass
    
    def on_release(key):
        try:
            char = key.char
            if char in ['y', 'p', 'q']:
                with keys_lock:
                    current_keys_pressed.discard(char)
                    print(f"[DEBUG] Key '{char}' released and removed from set")
                if char == 'q':
                    print("[CALIBRATION] Quit requested")
                    return False  # Stop listener
        except AttributeError:
            pass
    
    try:
        listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        while True:
            try:
                time.sleep(0.1)
                with keys_lock:
                    if 'q' in current_keys_pressed:
                        print("[CALIBRATION] Quit requested")
                        listener.stop()
                        break
            except Exception as e:
                print(f"[CALIBRATION] Error monitoring keys: {e}")
                break
                
    except Exception as e:
        print(f"[CALIBRATION] pynput error: {e}")
    
    print("[CALIBRATION] Keyboard handler stopped")


def is_key_pressed(key_char):
    if not KEYBOARD_AVAILABLE:
        return False
    
    with keys_lock:
        is_pressed = key_char in current_keys_pressed
        if is_pressed:
            print(f"[DEBUG] Key '{key_char}' detected as pressed!")
        return is_pressed


class CameraIntrinsics:
    """Store and manage camera intrinsic parameters for depth-to-world coordinate conversion"""
    def __init__(self):
        self.fx = None  
        self.fy = None  
        self.ppx = None 
        self.ppy = None  
        self.intrinsics_valid = False
        
    def set_intrinsics(self, fx, fy, ppx, ppy):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        self.intrinsics_valid = True

camera_intrinsics = CameraIntrinsics()


def pixel_to_world_coordinates(pixel_x, pixel_y, depth_mm, intrinsics_data=None):    
    # Use provided intrinsics_data if available, otherwise fall back to global
    if intrinsics_data is not None:
        fx, fy, ppx, ppy = intrinsics_data['fx'], intrinsics_data['fy'], intrinsics_data['ppx'], intrinsics_data['ppy']
        intrinsics_valid = True
    else:
        fx, fy, ppx, ppy = camera_intrinsics.fx, camera_intrinsics.fy, camera_intrinsics.ppx, camera_intrinsics.ppy
        intrinsics_valid = camera_intrinsics.intrinsics_valid
    
    if not intrinsics_valid or depth_mm <= 0:
        return None, None, None
        
    try:
        # Convert depth from mm to meters
        depth_m = depth_mm / 1000.0
        
        # Convert pixel coordinates to world coordinates using camera intrinsics
        world_x = (pixel_x - ppx) * depth_m / fx
        world_y = (pixel_y - ppy) * depth_m / fy
        world_z = depth_m
        
        return world_x, world_y, world_z
        
    except Exception as e:
        return None, None, None


def landmark_to_world_coordinates(keypoint_2d, depth_frame, landmark_idx, intrinsics_data=None):
    """
    This function converts the input 2d landmarks in 3d world coordinates
    """
    if (depth_frame is None or keypoint_2d is None or 
        len(keypoint_2d.landmark) <= landmark_idx):
        return None, None, None
        
    try:
        # Get 2D landmark coordinates  
        landmark = keypoint_2d.landmark[landmark_idx]
        pixel_x = landmark.x * depth_frame.shape[1] 
        pixel_y = landmark.y * depth_frame.shape[0]
        
        # Get depth at landmark location
        x_int = int(np.clip(pixel_x, 0, depth_frame.shape[1] - 1))
        y_int = int(np.clip(pixel_y, 0, depth_frame.shape[0] - 1))
        depth_mm = float(depth_frame[y_int, x_int])
        
        if depth_mm <= 0:
            return None, None, None
            
        # Convert to world coordinates
        world_x, world_y, world_z = pixel_to_world_coordinates(pixel_x, pixel_y, depth_mm, intrinsics_data)
        
        return world_x, world_y, world_z
        
    except Exception as e:
        return None, None, None


def extract_camera_intrinsics_from_data_package(data_package):
    if 'intrinsics' in data_package:
        intrinsics_data = data_package['intrinsics']
        camera_intrinsics.set_intrinsics(
            intrinsics_data['fx'], 
            intrinsics_data['fy'],
            intrinsics_data['ppx'], 
            intrinsics_data['ppy']
        )

def calculate_single_finger_angles(joint_pos, finger_landmarks):
    """
    Calculates MCP, PIP, and DIP angles plus spread for a single finger.
    """
    if joint_pos is None:
        return None
        
    try:
        wrist_pos = joint_pos[0] 
        mcp_pos = joint_pos[finger_landmarks[0]]  
        pip_pos = joint_pos[finger_landmarks[1]]  
        dip_pos = joint_pos[finger_landmarks[2]]  
        tip_pos = joint_pos[finger_landmarks[3]]  
        
        mcp_to_pip_vec = pip_pos - mcp_pos
        mcp_to_pip_vec = mcp_to_pip_vec / np.linalg.norm(mcp_to_pip_vec)
        
        pip_to_dip_vec = dip_pos - pip_pos
        pip_to_dip_vec = pip_to_dip_vec / np.linalg.norm(pip_to_dip_vec)
        
        dip_to_tip_vec = tip_pos - dip_pos
        dip_to_tip_vec = dip_to_tip_vec / np.linalg.norm(dip_to_tip_vec)
        
        wrist_to_mcp_vec = mcp_pos - wrist_pos
        wrist_to_mcp_vec = wrist_to_mcp_vec / np.linalg.norm(wrist_to_mcp_vec)
        
        mcp_flex_angle = np.pi - np.arccos(np.clip(np.dot(wrist_to_mcp_vec, mcp_to_pip_vec), -1, 1))
        pip_flex_angle = np.pi - np.arccos(np.clip(np.dot(mcp_to_pip_vec, pip_to_dip_vec), -1, 1))
        dip_flex_angle = np.pi - np.arccos(np.clip(np.dot(pip_to_dip_vec, dip_to_tip_vec), -1, 1))
        
        mcp_spread_angle = np.arctan2(mcp_to_pip_vec[1], np.sqrt(mcp_to_pip_vec[0]**2 + mcp_to_pip_vec[2]**2))
        
        return {
            'mcp_spread': mcp_spread_angle,
            'mcp_flex': mcp_flex_angle,
            'pip_flex': pip_flex_angle,
            'dip_flex': dip_flex_angle
        }
        
    except Exception as e:
        return None


def calculate_finger_angles_for_finger_enhanced(joint_pos, finger_landmarks, keypoint_2d, depth_frame, intrinsics_data=None):
    """
    Calculates MCP, PIP, and DIP angles plus spread for a single finger.
    """
    if joint_pos is None:
        return None
        
    try:
        # Try to get enhanced 3D positions using actual depth data (all landmarks needed)
        enhanced_positions = {}
        landmark_names = ['wrist', 'mcp', 'pip', 'dip', 'tip']  
        landmark_indices = [0] + finger_landmarks  
        
        for i, (name, idx) in enumerate(zip(landmark_names, landmark_indices)):
            world_x, world_y, world_z = landmark_to_world_coordinates(keypoint_2d, depth_frame, idx, intrinsics_data)
            if world_x is not None:
                enhanced_positions[name] = np.array([world_x, world_y, world_z])
            else:
                # Fallback to MediaPipe inferred 3D position
                enhanced_positions[name] = joint_pos[idx]
                
        # Use enhanced positions for calculations
        wrist_pos = enhanced_positions['wrist']
        mcp_pos = enhanced_positions['mcp']
        pip_pos = enhanced_positions['pip'] 
        dip_pos = enhanced_positions['dip']
        tip_pos = enhanced_positions['tip']
        
        mcp_to_pip_vec = pip_pos - mcp_pos
        mcp_to_pip_vec = mcp_to_pip_vec / np.linalg.norm(mcp_to_pip_vec)
        
        pip_to_dip_vec = dip_pos - pip_pos
        pip_to_dip_vec = pip_to_dip_vec / np.linalg.norm(pip_to_dip_vec)
        
        dip_to_tip_vec = tip_pos - dip_pos
        dip_to_tip_vec = dip_to_tip_vec / np.linalg.norm(dip_to_tip_vec)
        
        wrist_to_mcp_vec = mcp_pos - wrist_pos
        wrist_to_mcp_vec = wrist_to_mcp_vec / np.linalg.norm(wrist_to_mcp_vec)
        
        mcp_flex_angle = np.pi - np.arccos(np.clip(np.dot(wrist_to_mcp_vec, mcp_to_pip_vec), -1, 1))
        pip_flex_angle = np.pi - np.arccos(np.clip(np.dot(mcp_to_pip_vec, pip_to_dip_vec), -1, 1))
        dip_flex_angle = np.pi - np.arccos(np.clip(np.dot(pip_to_dip_vec, dip_to_tip_vec), -1, 1))
        
        mcp_spread_angle = np.arctan2(mcp_to_pip_vec[1], np.sqrt(mcp_to_pip_vec[0]**2 + mcp_to_pip_vec[2]**2))
        
        depth_used_count = sum(1 for pos in enhanced_positions.values() if isinstance(pos, np.ndarray) and len(pos) == 3)
        
        return {
            'mcp_spread': mcp_spread_angle,
            'mcp_flex': mcp_flex_angle,
            'pip_flex': pip_flex_angle,
            'dip_flex': dip_flex_angle,
            'depth_enhanced_count': depth_used_count  
        }
        
    except Exception as e:
        # Fallback to original method
        return calculate_single_finger_angles(joint_pos, finger_landmarks)


def calculate_direct_finger_angles_depth_enhanced(joint_pos, keypoint_2d, depth_frame, intrinsics_data=None):
    """ 
    Returns:
        dict: Joint angles in radians for all fingers
        
        NOTE: For joint calculations, we ignore intrinsics_data and use MediaPipe 3D positions
        to maintain consistency with rgb_retargetting.py in joints value. 
        The intrinsics_data is only used for world coordinate calculations (X,Y,Z).
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
        total_depth_enhanced = 0
        
        # Calculate angles for each finger using MediaPipe positions only (no depth enhancement)
        # This ensures we get the same joint values as joint_receiver_example - Copie.py
        for finger_name, landmarks in finger_landmarks.items():
            finger_angles = calculate_single_finger_angles(joint_pos, landmarks)
            if finger_angles is not None:
                all_finger_angles[finger_name] = finger_angles
        
        # For backward compatibility, return index finger angles
        if 'index' in all_finger_angles:
            result = all_finger_angles['index'].copy()
            result['all_fingers'] = all_finger_angles
            result['total_depth_enhanced'] = 0  # Always 0 since we're not using depth for joints
            return result
        else:
            return None
        
    except Exception as e:
        # Fallback to original method
        return calculate_direct_finger_angles(joint_pos)


def calculate_direct_finger_angles(joint_pos):
    """ 
    with no depth enhancement
    Returns:
        dict: Joint angles in radians for all fingers
    """
    if joint_pos is None:
        return None
        
    try:
        # Define finger landmark indices: [MCP, PIP, DIP, TIP]
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
        
        if 'index' in all_finger_angles:
            result = all_finger_angles['index'].copy()
            result['all_fingers'] = all_finger_angles
            return result
        else:
            return None
        
    except Exception as e:
        return None



def calculate_thumb_joint_angles_depth_enhanced(joint_pos, keypoint_2d, depth_frame, intrinsics_data=None):
    """ 
    Returns:
        dict: Thumb joint angles in degrees
        
        NOTE: For joint calculations, we ignore intrinsics_data and use MediaPipe 3D positions
        to maintain consistency with rgb_retargetting.py in joints value. 
        The intrinsics_data is only used for world coordinate calculations (X,Y,Z).
    """
    if joint_pos is None:
        return None
        
    try:
        # Use MediaPipe positions only (no depth enhancement) for consistent joint values
        wrist_pos = joint_pos[0]
        cmc_pos = joint_pos[1]
        mcp_pos = joint_pos[2]
        ip_pos = joint_pos[3]
        tip_pos = joint_pos[4]

        wrist_to_cmc_vec = cmc_pos - wrist_pos
        cmc_to_mcp_vec = mcp_pos - cmc_pos
        mcp_to_ip_vec = ip_pos - mcp_pos
        ip_to_tip_vec = tip_pos - ip_pos
        
        wrist_to_cmc_vec = wrist_to_cmc_vec / np.linalg.norm(wrist_to_cmc_vec)
        cmc_to_mcp_vec = cmc_to_mcp_vec / np.linalg.norm(cmc_to_mcp_vec)
        mcp_to_ip_vec = mcp_to_ip_vec / np.linalg.norm(mcp_to_ip_vec)
        ip_to_tip_vec = ip_to_tip_vec / np.linalg.norm(ip_to_tip_vec)
        
        cmc1_angle = np.arctan2(wrist_to_cmc_vec[1], np.sqrt(wrist_to_cmc_vec[0]**2 + wrist_to_cmc_vec[2]**2))
        cmc2_angle = np.pi - np.arccos(np.clip(np.dot(wrist_to_cmc_vec, cmc_to_mcp_vec), -1, 1))
        mcp_angle = np.pi - np.arccos(np.clip(np.dot(cmc_to_mcp_vec, mcp_to_ip_vec), -1, 1))
        ip_angle = np.pi - np.arccos(np.clip(np.dot(mcp_to_ip_vec, ip_to_tip_vec), -1, 1))
        
        # Convert to degrees
        return {
            'cmc1_deg': np.degrees(cmc1_angle),
            'cmc2_deg': np.degrees(cmc2_angle),
            'mcp_deg': np.degrees(mcp_angle),
            'ip_deg': np.degrees(ip_angle),
            'depth_enhanced_count': 0  # Always 0 since we're not using depth for joints
        }
        
    except Exception as e:
        # Fallback to original method
        return calculate_thumb_joint_angles(joint_pos)


def calculate_thumb_joint_angles(joint_pos):
    """ 
    with no depth enhancement
    Returns:
        dict: Thumb joint angles in degrees
    """
    if joint_pos is None:
        return None
        
    try:
        wrist_pos = joint_pos[0]
        cmc_pos = joint_pos[1]
        mcp_pos = joint_pos[2]
        ip_pos = joint_pos[3]
        tip_pos = joint_pos[4]

        wrist_to_cmc_vec = cmc_pos - wrist_pos
        cmc_to_mcp_vec = mcp_pos - cmc_pos
        mcp_to_ip_vec = ip_pos - mcp_pos
        ip_to_tip_vec = tip_pos - ip_pos
        
        wrist_to_cmc_vec = wrist_to_cmc_vec / np.linalg.norm(wrist_to_cmc_vec)
        cmc_to_mcp_vec = cmc_to_mcp_vec / np.linalg.norm(cmc_to_mcp_vec)
        mcp_to_ip_vec = mcp_to_ip_vec / np.linalg.norm(mcp_to_ip_vec)
        ip_to_tip_vec = ip_to_tip_vec / np.linalg.norm(ip_to_tip_vec)
        
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
        
        # For backward compatibility, return index finger angles 
        if 'index' in all_finger_angles:
            result = all_finger_angles['index'].copy()
            result['all_fingers'] = all_finger_angles
            return result
        else:
            return None
        
    except Exception as e:
        return None


def finger_mcp_spread(joint_pos):
    """
    Calculate the average angle (in radians) between MCP-PIP vectors of adjacent fingers:
    - Index-Middle
    - Middle-Ring
    - Ring-Pinky
    Returns average of these three angles.
    """
    if joint_pos is None:
        return None

    try:
        mcp_indices = {
            'index': 5,
            'middle': 9,
            'ring': 13,
            'pinky': 17
        }
        pip_indices = {
            'index': 6,
            'middle': 10,
            'ring': 14,
            'pinky': 18
        }

        # Compute MCP-PIP vectors for each finger
        mcp_pip_vecs = {}
        for finger in ['index', 'middle', 'ring', 'pinky']:
            mcp = joint_pos[mcp_indices[finger]]
            pip = joint_pos[pip_indices[finger]]
            vec = pip - mcp
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None
            mcp_pip_vecs[finger] = vec / norm

        angles = []
        # Index-Middle
        dot_im = np.clip(np.dot(mcp_pip_vecs['index'], mcp_pip_vecs['middle']), -1.0, 1.0)
        angle_im = np.arccos(dot_im)
        angles.append(angle_im)
        # Middle-Ring
        dot_mr = np.clip(np.dot(mcp_pip_vecs['middle'], mcp_pip_vecs['ring']), -1.0, 1.0)
        angle_mr = np.arccos(dot_mr)
        angles.append(angle_mr)
        # Ring-Pinky
        dot_rp = np.clip(np.dot(mcp_pip_vecs['ring'], mcp_pip_vecs['pinky']), -1.0, 1.0)
        angle_rp = np.arccos(dot_rp)
        angles.append(angle_rp)

        average_angle = np.mean(angles)
        return average_angle

    except Exception as e:
        return None



def map_finger_spread_to_degrees(pip_distance):
    """
    Maps the actual range [0.08, 0.14] to [-120, +120] degrees.
    """
    if pip_distance is None:
        return None
    
    try:
        spread_normalized = (pip_distance - 0.08) / (0.14 - 0.08)  # 0 = closed, 1 = spread
        spread_normalized = np.clip(spread_normalized, 0.0, 1.0)
        spread_mapped = -120.0 + (spread_normalized * 240.0)  # -120 to +120 degrees
        
        return np.clip(spread_mapped, -120.0, 120.0)
        
    except Exception as e:
        return None


def map_index_joints(mcp_flex_deg, pip_flex_deg):
    """
    Map direct finger angle values from RGBd retargeting to hand joint positions.
    
    Updated ranges based on testing:
    - MCP Flex range: +120° to +165° (where +165° = straight, +120° = fully closed) → 0 to 200
    - PIP Flex range: +120° to +165° (where +165° = straight, +120° = fully closed) → 0 to 200
    """
    mapped_index_joints = {}
    
    if mcp_flex_deg is not None:
        # Map MCP flexion: +120° to +165° range (165° = straight, 120° = closed)
        flex_normalized = (165.0 - mcp_flex_deg) / (165.0 - 120.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        mcp_joint_value = flex_normalized * (200)  # 0 to 200 rad
        mapped_index_joints["Index_MCP"] = np.clip(mcp_joint_value, 0.0, 200)
    
    if pip_flex_deg is not None:
        # Map PIP flexion: +120° to +165° range (165° = straight, 120° = fully closed)
        flex_normalized = (165.0 - pip_flex_deg) / (165.0 - 120.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        pip_joint_value = flex_normalized * (200)  # 0 to 200 rad
        mapped_index_joints["Index_PIP"] = np.clip(pip_joint_value, 0.0, 200)
    
    return mapped_index_joints


def map_middle_finger_joints(mcp_flex_deg, pip_flex_deg):
    """Map RGB retargeting middle finger angles to Isaac Lab joint positions."""
    mapped_middle_joints = {}
    
    if mcp_flex_deg is not None:
        # Map MCP flexion: 150° (straight) to 110° (closed)
        flex_normalized = (150.0 - mcp_flex_deg) / (150.0 - 110.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        mcp_joint_value = flex_normalized * (200)  # 0 to 200 rad
        mapped_middle_joints["Middle_MCP"] = np.clip(mcp_joint_value, 0.0, 200)
    
    if pip_flex_deg is not None:
        # Map PIP flexion: 175° (straight) to 100° (closed)
        flex_normalized = (175.0 - pip_flex_deg) / (175.0 - 100.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        pip_joint_value = flex_normalized * (200)  # 0 to 200 rad
        mapped_middle_joints["Middle_PIP"] = np.clip(pip_joint_value, 0.0, 200)
    
    return mapped_middle_joints


def map_ring_finger_joints(mcp_flex_deg, pip_flex_deg):
    """Map RGB retargeting ring finger angles to Isaac Lab joint positions."""
    mapped_ring_joints = {}
    
    if mcp_flex_deg is not None:
        # Map MCP flexion: 130° (straight) to 110° (closed)
        flex_normalized = (130.0 - mcp_flex_deg) / (130.0 - 110.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        mcp_joint_value = flex_normalized * (200)  # 0 to 200 rad
        mapped_ring_joints["Ring_MCP"] = np.clip(mcp_joint_value, 0.0, 200)
    
    if pip_flex_deg is not None:
        # Map PIP flexion: 170° (straight) to 100° (closed)
        flex_normalized = (170.0 - pip_flex_deg) / (170.0 - 100.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        pip_joint_value = flex_normalized * (200)  # 0 to 200 rad
        mapped_ring_joints["Ring_PIP"] = np.clip(pip_joint_value, 0.0, 200)
    
    return mapped_ring_joints


def map_pinky_finger_joints(mcp_flex_deg, pip_flex_deg):
    """Map RGB retargeting pinky finger angles to Isaac Lab joint positions."""
    mapped_pinky_joints = {}
    
    if mcp_flex_deg is not None:
        # Map MCP flexion: 140° (straight) to 100° (closed)
        flex_normalized = (140.0 - mcp_flex_deg) / (140.0 - 100.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        mcp_joint_value = flex_normalized * (200)  # 0 to 200 rad
        mapped_pinky_joints["Pinky_MCP"] = np.clip(mcp_joint_value, 0.0, 200)
    
    if pip_flex_deg is not None:
        # Map PIP flexion: 170° (straight) to 110° (closed)
        flex_normalized = (170.0 - pip_flex_deg) / (170.0 - 110.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        pip_joint_value = flex_normalized * (200)  # 0 to 200 rad
        mapped_pinky_joints["Pinky_PIP"] = np.clip(pip_joint_value, 0.0, 200)
    return mapped_pinky_joints


def map_thumb_joints(cmc1_deg, cmc2_deg, mcp_deg, ip_deg):
    """Map RGB retargeting thumb angles to Isaac Lab joint positions."""
    mapped_thumb_joints = {}
    
    if cmc1_deg is not None:
        # Map CMC1: 60° = 0, 25° = 200
        cmc1_clamped = max(25.0, min(60.0, cmc1_deg))
        cmc1_normalized = (60.0 - cmc1_clamped) / (60.0 - 25.0)  # 0 = 60°, 1 = 25°
        cmc1_joint_value = cmc1_normalized * (200)  # 0.0 to 200 range
        mapped_thumb_joints["Thumb_CMC1"] = np.clip(cmc1_joint_value, 0.0, 200)
    
    if cmc2_deg is not None:
        # Map CMC2: 165° = 0, 135° = 200
        cmc2_clamped = max(135.0, min(165.0, cmc2_deg))
        cmc2_normalized = (165.0 - cmc2_clamped) / (165.0 - 135.0)  # 0 = 165°, 1 = 135°
        cmc2_joint_value = cmc2_normalized * (200)  # 0.0 to 200 range
        final_value = np.clip(cmc2_joint_value, 0.0, 200)
        mapped_thumb_joints["Thumb_CMC2"] = final_value
    
    if mcp_deg is not None:
        # Map MCP: 170° = 0, 140° = 200
        mcp_clamped = max(140.0, min(170.0, mcp_deg))
        mcp_normalized = (170.0 - mcp_clamped) / (170.0 - 140.0)  # 0 = 170°, 1 = 140°
        mcp_joint_value = mcp_normalized * (200)  # 0.0 to 200 range
        mapped_thumb_joints["Thumb_MCP"] = np.clip(mcp_joint_value, 0.0, 200)
    
    if ip_deg is not None:
        # Map IP: 175° = 0, 130° = 200
        ip_clamped = max(130.0, min(175.0, ip_deg))
        ip_normalized = (175.0 - ip_clamped) / (175.0 - 130.0)  # 0 = 175°, 1 = 130°
        ip_joint_value = ip_normalized * (200)  # 0.0 to 200 range
        mapped_thumb_joints["Thumb_IP"] = np.clip(ip_joint_value, 0.0, 200)
    
    return mapped_thumb_joints


def compute_yaw_pitch(rotation_matrix):
    global calibration_data, calibration_lock
    
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


def handle_calibration(yaw_raw_deg, pitch_raw_deg):
    """Handle keyboard-based calibration for yaw and pitch"""
    global calibration_data, calibration_lock
    
    with calibration_lock:
        try:
            if is_key_pressed('y'):
                if not calibration_data['yaw_calibrating']:
                    # Start calibration - set neutral position
                    calibration_data['yaw_calibrating'] = True
                    calibration_data['yaw_neutral'] = yaw_raw_deg
                    calibration_data['yaw_min'] = yaw_raw_deg
                    calibration_data['yaw_max'] = yaw_raw_deg
                    print(f"[CALIBRATION] Yaw calibration started. Neutral position set to {yaw_raw_deg:.2f}°")
                else:
                    # Update min/max during calibration
                    if yaw_raw_deg < calibration_data['yaw_min']:
                        calibration_data['yaw_min'] = yaw_raw_deg
                    if yaw_raw_deg > calibration_data['yaw_max']:
                        calibration_data['yaw_max'] = yaw_raw_deg
                    print(f"[CALIBRATION] Yaw range: {calibration_data['yaw_min']:.2f}° to {calibration_data['yaw_max']:.2f}° (current: {yaw_raw_deg:.2f}°)")
            else:
                if calibration_data['yaw_calibrating']:
                    # End calibration
                    calibration_data['yaw_calibrating'] = False
                    print(f"[CALIBRATION] Yaw calibration complete. Range: {calibration_data['yaw_min']:.2f}° to {calibration_data['yaw_max']:.2f}°")
            
            if is_key_pressed('p'):
                if not calibration_data['pitch_calibrating']:
                    # Start calibration - set neutral position
                    calibration_data['pitch_calibrating'] = True
                    calibration_data['pitch_neutral'] = pitch_raw_deg
                    calibration_data['pitch_min'] = pitch_raw_deg
                    calibration_data['pitch_max'] = pitch_raw_deg
                    print(f"[CALIBRATION] Pitch calibration started. Neutral position set to {pitch_raw_deg:.2f}°")
                else:
                    # Update min/max during calibration
                    if pitch_raw_deg < calibration_data['pitch_min']:
                        calibration_data['pitch_min'] = pitch_raw_deg
                    if pitch_raw_deg > calibration_data['pitch_max']:
                        calibration_data['pitch_max'] = pitch_raw_deg
                    print(f"[CALIBRATION] Pitch range: {calibration_data['pitch_min']:.2f}° to {calibration_data['pitch_max']:.2f}° (current: {pitch_raw_deg:.2f}°)")
            else:
                if calibration_data['pitch_calibrating']:
                    # End calibration
                    calibration_data['pitch_calibrating'] = False
                    print(f"[CALIBRATION] Pitch calibration complete. Range: {calibration_data['pitch_min']:.2f}° to {calibration_data['pitch_max']:.2f}°")
        except Exception as e:
            # Keyboard library not available or other error - skip calibration
            pass


def map_yaw_pitch(yaw_raw_deg=0.0, pitch_raw_deg=0.0):
    """
    Map RGBd retargeting yaw/pitch values to adapt hand joint positions for wrist orientation.
    Uses calibrated values if available, otherwise falls back to default ranges.
    
    Default specifications (when not calibrated):
    - Straight hand position: yaw = 0°, pitch = 0° (raw values)
    - Yaw range: -25° (left) to +25° (right)  
    - Pitch range: -30° (up) to +30° (down)
    """
    global calibration_data, calibration_lock
    
    mapped_joints = {}
    
    with calibration_lock:
        # Use calibrated yaw mapping if available
        if (calibration_data['yaw_min'] is not None and 
            calibration_data['yaw_max'] is not None and
            calibration_data['yaw_min'] != calibration_data['yaw_max']):
            
            # Map using calibrated range - neutral maps to 0, extremes map to ±120
            yaw_deviation = yaw_raw_deg - calibration_data['yaw_neutral']
            max_positive_deviation = calibration_data['yaw_max'] - calibration_data['yaw_neutral']
            max_negative_deviation = calibration_data['yaw_min'] - calibration_data['yaw_neutral']
            
            if yaw_deviation >= 0:
                # Positive deviation: map to 0 to +120
                if max_positive_deviation > 0:
                    yaw_normalized = yaw_deviation / max_positive_deviation  # 0 to 1
                    wrist_yaw_mapped = yaw_normalized * 120.0  # 0 to +120
                else:
                    wrist_yaw_mapped = 0.0
            else:
                # Negative deviation: map to 0 to -120
                if max_negative_deviation < 0:
                    yaw_normalized = yaw_deviation / max_negative_deviation  # 0 to 1 (for negative values)
                    wrist_yaw_mapped = yaw_normalized * -120.0  # 0 to -120
                else:
                    wrist_yaw_mapped = 0.0
            
            # Clamp to valid range
            wrist_yaw_mapped = np.clip(wrist_yaw_mapped, -120.0, 120.0)
            
            if calibration_data['yaw_calibrating']:
                print(f"[CALIBRATION] Yaw: {yaw_raw_deg:.2f}° (dev: {yaw_deviation:.2f}°) → {wrist_yaw_mapped:.2f}° (neutral: {calibration_data['yaw_neutral']:.2f}°)")
        else:
            # Use default yaw mapping for raw values
            yaw_clamped = max(-25.0, min(25.0, yaw_raw_deg))
            yaw_normalized = (yaw_clamped + 25.0) / 50.0  # 0 to 1 range
            wrist_yaw_mapped = -120.0 + (yaw_normalized * 240.0)  # -120 to +120 degrees
        
        mapped_joints["Wrist_Yaw"] = wrist_yaw_mapped

        # Use calibrated pitch mapping if available
        if (calibration_data['pitch_min'] is not None and 
            calibration_data['pitch_max'] is not None and
            calibration_data['pitch_min'] != calibration_data['pitch_max']):
            
            # Map using calibrated range - neutral maps to 0, extremes map to ±120
            pitch_deviation = pitch_raw_deg - calibration_data['pitch_neutral']
            max_positive_deviation = calibration_data['pitch_max'] - calibration_data['pitch_neutral']
            max_negative_deviation = calibration_data['pitch_min'] - calibration_data['pitch_neutral']
            
            if pitch_deviation >= 0:
                # Positive deviation: map to 0 to +120
                if max_positive_deviation > 0:
                    pitch_normalized = pitch_deviation / max_positive_deviation  # 0 to 1
                    wrist_pitch_mapped = pitch_normalized * 120.0  # 0 to +120
                else:
                    wrist_pitch_mapped = 0.0
            else:
                # Negative deviation: map to 0 to -120
                if max_negative_deviation < 0:
                    pitch_normalized = pitch_deviation / max_negative_deviation  # 0 to 1 (for negative values)
                    wrist_pitch_mapped = pitch_normalized * -120.0  # 0 to -120
                else:
                    wrist_pitch_mapped = 0.0
            
            # Clamp to valid range
            wrist_pitch_mapped = np.clip(wrist_pitch_mapped, -120.0, 120.0)
            
            if calibration_data['pitch_calibrating']:
                print(f"[CALIBRATION] Pitch: {pitch_raw_deg:.2f}° (dev: {pitch_deviation:.2f}°) → {wrist_pitch_mapped:.2f}° (neutral: {calibration_data['pitch_neutral']:.2f}°)")
        else:
            # Use default pitch mapping for raw values
            pitch_clamped = max(-30.0, min(30.0, pitch_raw_deg))
            pitch_normalized = (pitch_clamped + 30.0) / 60.0  # 0 to 1 range
            wrist_pitch_mapped = -120.0 + (pitch_normalized * 240.0)  # -120 to +120 degrees
        
        mapped_joints["Wrist_Pitch"] = wrist_pitch_mapped

    return mapped_joints


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
    MediaPipe-based hand tracking system that detects hand landmarks and converts them to MANO convetion.
    
    Processes RGB camera frames to extract 21 hand landmarks, estimates 3D joint positions,
    computes hand orientation matrix.
    Supports both left and right hand tracking with configurable detection confidence.
    """
    
    # Hand orientation matrices (copied from dex-retargeting)
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

    def detect(self, rgb_image):
        """
        Detect hand landmarks and return processed data.
        
        Returns:
            tuple: (num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot, confidence)
        """
        results = self.hand_detector.process(rgb_image)
        
        if results.multi_hand_landmarks is None or results.multi_handedness is None:
            return 0, None, None, None, 0.0
            
        hand_confidence = results.multi_handedness[0].classification[0].score
            
        keypoint_3d = results.multi_hand_landmarks[0]
        keypoint_2d = results.multi_hand_landmarks[0]
        
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        
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
    'yaw_raw_deg': None,
    'pitch_raw_deg': None,
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
    'intrinsics_data': None,
    'timestamp': None,
    'finger_spread_mapped': None
}
joint_data_lock = threading.Lock()


def joint_data_server(host='0.0.0.0', port=8889):
    """Server to send hand data to Isaac Sim - Intel RealSense Source"""
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
                                
                                data_to_send = {
                                    'yaw_deg': latest_joint_data.get('yaw_deg', 0.0),
                                    'pitch_deg': latest_joint_data.get('pitch_deg', 0.0),
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
                                    'middle_mcp_flex_deg': latest_joint_data.get('middle_mcp_flex_deg'),
                                    'middle_pip_flex_deg': latest_joint_data.get('middle_pip_flex_deg'),
                                    'ring_mcp_flex_deg': latest_joint_data.get('ring_mcp_flex_deg'),
                                    'ring_pip_flex_deg': latest_joint_data.get('ring_pip_flex_deg'),
                                    'pinky_mcp_flex_deg': latest_joint_data.get('pinky_mcp_flex_deg'),
                                    'pinky_pip_flex_deg': latest_joint_data.get('pinky_pip_flex_deg'),
                                    'thumb_cmc1_deg': latest_joint_data.get('thumb_cmc1_deg'),
                                    'thumb_cmc2_deg': latest_joint_data.get('thumb_cmc2_deg'),
                                    'thumb_mcp_deg': latest_joint_data.get('thumb_mcp_deg'),
                                    'thumb_ip_deg': latest_joint_data.get('thumb_ip_deg'),
                                    'confidence': latest_joint_data.get('confidence', 0.0),
                                    'timestamp': latest_joint_data.get('timestamp')
                                }
                                
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

    server_thread = threading.Thread(target=joint_data_server, daemon=True)
    server_thread.start()
    
    joint_client_thread = threading.Thread(target=start_joint_data_client, daemon=True)
    joint_client_thread.start()

    calibration_thread = threading.Thread(target=keyboard_calibration_handler, daemon=True)
    calibration_thread.start()

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            return

        current_depth_frame = None
        current_intrinsics_data = None
        try:
            while not depth_queue.empty():
                depth_package = depth_queue.get_nowait()
                if isinstance(depth_package, dict):
                    current_depth_frame = depth_package.get('depth_frame')
                    current_intrinsics_data = depth_package.get('intrinsics_data')
                else:
                    # Handle legacy format (just depth frame)
                    current_depth_frame = depth_package
                    current_intrinsics_data = None
        except:
            pass

        _, joint_pos, keypoint_2d, wrist_rot_matrix, hand_confidence = detector.detect(rgb)

        if joint_pos is None:
            with joint_data_lock:
                latest_joint_data['yaw_deg'] = None
                latest_joint_data['pitch_deg'] = None
                latest_joint_data['roll_deg'] = None
                latest_joint_data['confidence'] = 0.0  # No hand detected
                latest_joint_data['timestamp'] = time.time()  # Keep timestamp updated
        else:
            wrist_camera_x = None
            wrist_camera_y = None
            if keypoint_2d is not None and len(keypoint_2d.landmark) > 0:
                wrist_landmark = keypoint_2d.landmark[0]
                wrist_camera_x = wrist_landmark.x * 640
                wrist_camera_y = wrist_landmark.y * 480
            
            yaw_independent, pitch_independent = compute_yaw_pitch(wrist_rot_matrix)
            roll_v2 = compute_roll(wrist_rot_matrix)
            
            yaw_raw_deg = np.degrees(yaw_independent)
            pitch_raw_deg = np.degrees(pitch_independent)
            
            handle_calibration(yaw_raw_deg, pitch_raw_deg)
            
            yaw_deg = yaw_raw_deg
            pitch_deg = pitch_raw_deg
            roll_deg = np.degrees(roll_v2)
            
            wrist_world_x, wrist_world_y, wrist_world_z = None, None, None
            if wrist_camera_x is not None and wrist_camera_y is not None and current_depth_frame is not None:
                
                # Use the intrinsics_data from the depth queue (same as finger calculations)
                intrinsics_data = current_intrinsics_data
                                    
                x_pixel = int(np.clip(wrist_camera_x, 0, current_depth_frame.shape[1] - 1))
                y_pixel = int(np.clip(wrist_camera_y, 0, current_depth_frame.shape[0] - 1))
                wrist_depth_mm = float(current_depth_frame[y_pixel, x_pixel])
                
                # Debug depth around wrist position if depth is 0
                if wrist_depth_mm == 0.0:
                    print(f"[DEBUG] Depth is 0 at wrist position, checking nearby pixels")
                    # Check a 5x5 pixel area around the wrist
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            check_x = int(np.clip(x_pixel + dx, 0, current_depth_frame.shape[1] - 1))
                            check_y = int(np.clip(y_pixel + dy, 0, current_depth_frame.shape[0] - 1))
                            check_depth = float(current_depth_frame[check_y, check_x])
                            if check_depth > 0:
                                # Use the first valid depth found
                                wrist_depth_mm = check_depth
                                break
                        if wrist_depth_mm > 0:
                            break
                                    
                if intrinsics_data is not None and wrist_depth_mm > 0:
                    wrist_world_x, wrist_world_y, wrist_world_z = pixel_to_world_coordinates(
                        wrist_camera_x, wrist_camera_y, wrist_depth_mm, intrinsics_data
                    )
            else:
                print(f"[DEBUG] Missing data - camera_x: {wrist_camera_x}, camera_y: {wrist_camera_y}, depth_frame: {current_depth_frame is not None}")
            
            # Use intrinsics_data from the depth package instead of shared storage
            intrinsics_data = current_intrinsics_data
            
            direct_angles = calculate_direct_finger_angles_depth_enhanced(joint_pos, keypoint_2d, current_depth_frame, intrinsics_data)
            thumb_angles = calculate_thumb_joint_angles_depth_enhanced(joint_pos, keypoint_2d, current_depth_frame, intrinsics_data)
            
            intel_confidence = hand_confidence
            
            # Update global data
            with joint_data_lock:
                latest_joint_data['yaw_deg'] = yaw_deg
                latest_joint_data['pitch_deg'] = pitch_deg
                latest_joint_data['roll_deg'] = roll_deg
                latest_joint_data['yaw_raw_deg'] = yaw_raw_deg
                latest_joint_data['pitch_raw_deg'] = pitch_raw_deg
                latest_joint_data['wrist_camera_x'] = wrist_camera_x
                latest_joint_data['wrist_camera_y'] = wrist_camera_y
                latest_joint_data['wrist_world_x'] = wrist_world_x
                latest_joint_data['wrist_world_y'] = wrist_world_y
                latest_joint_data['wrist_world_z'] = wrist_world_z
                latest_joint_data['depth_frame'] = current_depth_frame
                latest_joint_data['timestamp'] = time.time()
                latest_joint_data['confidence'] = intel_confidence
                
                # Store finger angles
                if direct_angles is not None:
                    latest_joint_data['index_mcp_spread_deg'] = np.degrees(direct_angles['mcp_spread'])
                    latest_joint_data['index_mcp_flex_deg'] = np.degrees(direct_angles['mcp_flex'])
                    latest_joint_data['index_pip_flex_deg'] = np.degrees(direct_angles['pip_flex'])
                    latest_joint_data['index_dip_flex_deg'] = np.degrees(direct_angles['dip_flex'])
                    
                    if 'all_fingers' in direct_angles:
                        all_fingers = direct_angles['all_fingers']
                        
                        if 'middle' in all_fingers:
                            latest_joint_data['middle_mcp_spread_deg'] = np.degrees(all_fingers['middle']['mcp_spread'])
                            latest_joint_data['middle_mcp_flex_deg'] = np.degrees(all_fingers['middle']['mcp_flex'])
                            latest_joint_data['middle_pip_flex_deg'] = np.degrees(all_fingers['middle']['pip_flex'])
                            latest_joint_data['middle_dip_flex_deg'] = np.degrees(all_fingers['middle']['dip_flex'])
                        
                        if 'ring' in all_fingers:
                            latest_joint_data['ring_mcp_spread_deg'] = np.degrees(all_fingers['ring']['mcp_spread'])
                            latest_joint_data['ring_mcp_flex_deg'] = np.degrees(all_fingers['ring']['mcp_flex'])
                            latest_joint_data['ring_pip_flex_deg'] = np.degrees(all_fingers['ring']['pip_flex'])
                            latest_joint_data['ring_dip_flex_deg'] = np.degrees(all_fingers['ring']['dip_flex'])
                        
                        if 'pinky' in all_fingers:
                            latest_joint_data['pinky_mcp_spread_deg'] = np.degrees(all_fingers['pinky']['mcp_spread'])
                            latest_joint_data['pinky_mcp_flex_deg'] = np.degrees(all_fingers['pinky']['mcp_flex'])
                            latest_joint_data['pinky_pip_flex_deg'] = np.degrees(all_fingers['pinky']['pip_flex'])
                            latest_joint_data['pinky_dip_flex_deg'] = np.degrees(all_fingers['pinky']['dip_flex'])
                
                if thumb_angles is not None:
                    latest_joint_data['thumb_cmc1_deg'] = thumb_angles['cmc1_deg']
                    latest_joint_data['thumb_cmc2_deg'] = thumb_angles['cmc2_deg']
                    latest_joint_data['thumb_mcp_deg'] = thumb_angles['mcp_deg']
                    latest_joint_data['thumb_ip_deg'] = thumb_angles['ip_deg']
                
                avg_pip_distance = finger_mcp_spread(joint_pos)
                mapped_spread_degrees = None
                if avg_pip_distance is not None:
                    mapped_spread_degrees = map_finger_spread_to_degrees(avg_pip_distance)
                latest_joint_data['finger_spread_mapped'] = mapped_spread_degrees


def produce_frame_from_network(queue, depth_queue, host='127.0.0.1', port=9998):
    """Receive frames from network camera - adapted for Intel RealSense camera server"""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        
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
                    
                    # Extract camera intrinsics if available and store globally
                    intrinsics_data = None
                    if 'intrinsics' in data_package:
                        intrinsics_data = data_package['intrinsics']
                        extract_camera_intrinsics_from_data_package(data_package)
                    
                    # Store depth frame and intrinsics globally
                    global latest_joint_data, joint_data_lock
                    with joint_data_lock:
                        latest_joint_data['depth_frame'] = depth_frame
                        latest_joint_data['intrinsics_data'] = intrinsics_data
                    
                    # Put depth frame AND intrinsics in queue for consumer process
                    if not depth_queue.full():
                        depth_package = {
                            'depth_frame': depth_frame,
                            'intrinsics_data': intrinsics_data
                        }
                        depth_queue.put(depth_package)
                    
                    frame = color_frame
                else:
                    frame = data_package
                    with joint_data_lock:
                        latest_joint_data['depth_frame'] = None
                    
                    # Send empty depth package for legacy format
                    if not depth_queue.full():
                        depth_package = {
                            'depth_frame': None,
                            'intrinsics_data': None
                        }
                        depth_queue.put(depth_package)
            except Exception as e:
                frame = pickle.loads(frame_data)
                with joint_data_lock:
                    latest_joint_data['depth_frame'] = None
                
                # Send empty depth package for exception case
                if not depth_queue.full():
                    depth_package = {
                        'depth_frame': None,
                        'intrinsics_data': None
                    }
                    depth_queue.put(depth_package)

            # Put frame in queue
            if not queue.full():
                queue.put(frame)
                
    except Exception as e:
        pass
    finally:
        try:
            client_socket.close()
        except:
            pass


def get_joint_data():
    """
    Get all available joint data including mapped values
    
    Returns:
        dict: All joint angles in degrees with standardized naming and mapped values
    """
    with joint_data_lock:
        
        # Check if hand is detected
        hand_confidence = latest_joint_data.get('confidence', 0.0)
        
        if hand_confidence <= 0.0:
            # Return default values when no hand is detected (same as RGB file)
            return {
                'Index_MCP_mapped': 0.0,
                'Index_PIP_mapped': 0.0,
                'Middle_MCP_MCP_mapped': 0.0,
                'Middle_MCP_PIP_mapped': 0.0,
                'Ring_MCP_MCP_mapped': 0.0,
                'Ring_MCP_PIP_mapped': 0.0,
                'Pinky_MCP_mapped': 0.0,
                'Pinky_PIP_mapped': 0.0,
                'Thumb_CMC1_mapped': 0.0,
                'Thumb_CMC2_mapped': 0.0,
                'Thumb_MCP_mapped': 0.0,
                'Thumb_IP_mapped': 0.0,
                'Wrist_Yaw_mapped': 0.0,
                'Wrist_Pitch_mapped': 0.0,
                'Finger_Spread_mapped': 0.0,
                'confidence': 0.0,
            }
        
        mapped_index_joints = map_index_joints(
            latest_joint_data.get('index_mcp_flex_deg'),
            latest_joint_data.get('index_pip_flex_deg'),
        )

        mapped_middle_joints = map_middle_finger_joints(
            latest_joint_data.get('middle_mcp_flex_deg'),
            latest_joint_data.get('middle_pip_flex_deg'),
        )

        mapped_ring_joints = map_ring_finger_joints(
            latest_joint_data.get('ring_mcp_flex_deg'),
            latest_joint_data.get('ring_pip_flex_deg'),
        )

        mapped_pinky_joints = map_pinky_finger_joints(
            latest_joint_data.get('pinky_mcp_flex_deg'),
            latest_joint_data.get('pinky_pip_flex_deg'),
        )
        
        mapped_thumb_joints = map_thumb_joints(
            latest_joint_data.get('thumb_cmc1_deg'),
            latest_joint_data.get('thumb_cmc2_deg'),
            latest_joint_data.get('thumb_mcp_deg'),
            latest_joint_data.get('thumb_ip_deg'),
        )

        mapped_wrist = map_yaw_pitch(
            latest_joint_data.get('yaw_deg', 140.0),
            latest_joint_data.get('pitch_deg', -70.0)
        )

        finger_spread_mapped = latest_joint_data.get('finger_spread_mapped')
        
        return {
            # Raw angle values
            'Index_MCP_Spread': latest_joint_data.get('index_mcp_spread_deg'),
            'Index_MCP': latest_joint_data.get('index_mcp_flex_deg'),
            'Index_PIP': latest_joint_data.get('index_pip_flex_deg'),
            'Index_DIP': latest_joint_data.get('index_dip_flex_deg'),
            
            'Middle_MCP_Spread': latest_joint_data.get('middle_mcp_spread_deg'),
            'Middle_MCP': latest_joint_data.get('middle_mcp_flex_deg'),
            'Middle_PIP': latest_joint_data.get('middle_pip_flex_deg'),
            'Middle_DIP': latest_joint_data.get('middle_dip_flex_deg'),
            
            'Ring_MCP_Spread': latest_joint_data.get('ring_mcp_spread_deg'),
            'Ring_MCP': latest_joint_data.get('ring_mcp_flex_deg'),
            'Ring_PIP': latest_joint_data.get('ring_pip_flex_deg'),
            'Ring_DIP': latest_joint_data.get('ring_dip_flex_deg'),
            
            'Pinky_MCP_Spread': latest_joint_data.get('pinky_mcp_spread_deg'),
            'Pinky_MCP': latest_joint_data.get('pinky_mcp_flex_deg'),
            'Pinky_PIP': latest_joint_data.get('pinky_pip_flex_deg'),
            'Pinky_DIP': latest_joint_data.get('pinky_dip_flex_deg'),
            
            'Thumb_CMC1': latest_joint_data.get('thumb_cmc1_deg'),
            'Thumb_CMC2': latest_joint_data.get('thumb_cmc2_deg'),
            'Thumb_MCP': latest_joint_data.get('thumb_mcp_deg'),
            'Thumb_IP': latest_joint_data.get('thumb_ip_deg'),

            # Mapped values for Isaac Sim
            'Index_MCP_mapped': mapped_index_joints.get('Index_MCP', 0.0),
            'Index_PIP_mapped': mapped_index_joints.get('Index_PIP', 0.0),
            
            'Middle_MCP_MCP_mapped': mapped_middle_joints.get('Middle_MCP', 0.0),
            'Middle_MCP_PIP_mapped': mapped_middle_joints.get('Middle_PIP', 0.0),
            
            'Ring_MCP_MCP_mapped': mapped_ring_joints.get('Ring_MCP', 0.0),
            'Ring_MCP_PIP_mapped': mapped_ring_joints.get('Ring_PIP', 0.0),
            
            'Pinky_MCP_mapped': mapped_pinky_joints.get('Pinky_MCP', 0.0),
            'Pinky_PIP_mapped': mapped_pinky_joints.get('Pinky_PIP', 0.0),

            'Thumb_CMC1_mapped': mapped_thumb_joints.get('Thumb_CMC1', 0.0),
            'Thumb_CMC2_mapped': mapped_thumb_joints.get('Thumb_CMC2', 0.0),
            'Thumb_MCP_mapped': mapped_thumb_joints.get('Thumb_MCP', 0.0),
            'Thumb_IP_mapped': mapped_thumb_joints.get('Thumb_IP', 0.0),

            'Wrist_Yaw_mapped': mapped_wrist.get('Wrist_Yaw', 0.0),
            'Wrist_Pitch_mapped': mapped_wrist.get('Wrist_Pitch', 0.0),
            
            # Finger spread
            'Finger_Spread_mapped': finger_spread_mapped if finger_spread_mapped is not None else 0.0,
            
            # World coordinates
            'wrist_world_x': latest_joint_data.get('wrist_world_x'),
            'wrist_world_y': latest_joint_data.get('wrist_world_y'),
            'wrist_world_z': latest_joint_data.get('wrist_world_z'),
            'wrist_roll': latest_joint_data.get('roll_deg'),
            
            # Confidence value for source selection
            'confidence': hand_confidence,
        }


def start_joint_data_client(target_host='localhost', target_port=8893):
    """Send joint data to another application via TCP"""
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((target_host, target_port))
            print(f"Joint data client connected to {target_host}:{target_port}")
            
            while True:
                joint_data = get_joint_data()
                # Always send data with confidence value for command_sender to decide
                joint_data['timestamp'] = time.time()
                message = json.dumps(joint_data) + '\n'
                client_socket.send(message.encode())
                
                time.sleep(1/30)  # 30 FPS
                
        except (ConnectionRefusedError, BrokenPipeError, OSError):
            print(f"Joint data client: Connection failed, retrying in 5 seconds...")
            time.sleep(5)  # Retry connection
        except Exception as e:
            print(f"Joint data client error: {e}")
            time.sleep(1)
        finally:
            try:
                client_socket.close()
            except:
                pass


def main(hand_type="Right", camera_host="127.0.0.1", camera_port=9998):
    """
    Standalone hand tracker
    
    Args:
        hand_type: "Right" or "Left"
        camera_host: Camera server IP
        camera_port: Camera server port (9998 for Intel RealSense camera server)
    """
    print(f"Intel RealSense Camera Server: {camera_host}:{camera_port}")
    print(f"Detecting {hand_type} hand")
    print(f"Standalone hand tracker - Works with Intel RealSense camera server")
    print("-" * 80)

    queue = multiprocessing.Queue(maxsize=50)
    depth_queue = multiprocessing.Queue(maxsize=20)
    
    producer_process = multiprocessing.Process(
        target=produce_frame_from_network, args=(queue, depth_queue, camera_host, camera_port)
    )
    consumer_process = multiprocessing.Process(
        target=start_hand_tracking, args=(queue, depth_queue, hand_type)
    )

    producer_process.start()
    consumer_process.start()

    try:
        producer_process.join()
        consumer_process.join()
    except KeyboardInterrupt:
        producer_process.terminate()
        consumer_process.terminate()
        
        time.sleep(1)
        
        if producer_process.is_alive():
            producer_process.kill()
        if consumer_process.is_alive():
            consumer_process.kill()
    
    time.sleep(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Hand Tracker for Isaac Sim")
    parser.add_argument("--hand-type", choices=["Right", "Left"], default="Right", help="Hand to track")
    parser.add_argument("--camera-host", default="127.0.0.1", help="Camera server host")
    parser.add_argument("--camera-port", type=int, default=9998, help="Camera server port (9998 for Intel RealSense)")
    
    args = parser.parse_args()
    main(args.hand_type, args.camera_host, args.camera_port)
