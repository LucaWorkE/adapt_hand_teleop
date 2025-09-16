#!/usr/bin/env python3
"""
Unified Roll Data Receiver with Isaac Lab Controller

This script receives data from hand tracking retargeting source and launches Isaac Lab simulation:
1. show_realtime_retargeting_new_copy.py (sends data via TCP server on port 8888)

Uses RGB retargeting control method for hand control in Isaac Lab simulation.

Author: GitHub Copilot
Date: August 31, 2025
"""

import argparse
import socket
import json
import threading
import time
import math
from typing import Optional, Dict, Any
import logging

import torch
import numpy as np

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def design_scene():
    """Designs the scene by spawning ground plane, light, and the UR5 with Adapt Hand."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")
    
    # Create articulation configuration for the UR5 with Adapt Hand
    ur5_cfg = ArticulationCfg(
        prim_path="/World/Objects/UR5_AdaptHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=r"c:/Users/lucad/Desktop/ur5_adapt_hand_converted.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=16,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.05),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.2,
                "elbow_joint": 1.8,
                "wrist_1_joint": -0.6,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                "Wrist_Yaw": 0.0,
                "Wrist_Pitch": 0.0,
                "Index_MCP_Spread": -0.1, 
                "Index_MCP": -0.5,
                "Index_PIP": -0.5,
                "Index_DIP": -1.0,         
                "Middle_MCP": -0.5,
                "Middle_PIP": -0.5,
                "Pinky_MCP": -0.5,
                "Pinky_PIP": -0.5,
                "Ring_MCP": -0.5,
                "Ring_PIP": -0.5,
                "Thumb_MCP": -0.5,
                "Thumb_CMC1": 0.0,
                "Thumb_CMC2": -0.5,
                "Thumb_IP": -0.5,
            },
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*", "elbow_.*", "wrist_.*"],
                effort_limit_sim=80.0,
                velocity_limit_sim=1.5,
                stiffness=250.0,
                damping=80.0,
            ),
            "hand_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_MCP", ".*_PIP", ".*_DIP", ".*_IP", ".*_Yaw", ".*_Pitch"],
                effort_limit_sim=20.0,
                velocity_limit_sim=4, 
                stiffness=80.0,          
                damping=40.0,            
            ),

            "thumb_joints": ImplicitActuatorCfg(
                joint_names_expr=["Thumb_.*"], 
                effort_limit_sim=15.0,   
                velocity_limit_sim=2.0,  
                stiffness=40.0,          
                damping=35.0,            
            ),
        },
    )
    
    return Articulation(cfg=ur5_cfg)


parser = argparse.ArgumentParser(description="Unified Isaac Lab controller with confidence-based switching.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Global variable to store received retargeting data for hand control
received_retargeting_data = {
    'joint_positions': None,
    'wrist_position': None,
    'wrist_orientation': None,
    'adapt_qpos': None,
    'yaw_deg': None,
    'pitch_deg': None,
    'roll_deg': None,
    'index_mcp_spread_deg': None,
    'index_mcp_flex_deg': None,
    'index_pip_flex_deg': None,
    'index_dip_flex_deg': None,
    'middle_mcp_flex_deg': None,
    'middle_pip_flex_deg': None,
    'ring_mcp_flex_deg': None,
    'ring_pip_flex_deg': None,
    'pinky_mcp_flex_deg': None,
    'pinky_pip_flex_deg': None,
    'thumb_cmc1_deg': None,
    'thumb_cmc2_deg': None,
    'thumb_mcp_deg': None,
    'thumb_ip_deg': None,
    'connected': False,
    'timestamp': None
}
data_lock = threading.Lock()

class DualSourceDataReceiver:
    def __init__(self):
        self.rgb_data = {
            'yaw_deg': None,
            'pitch_deg': None, 
            'roll_deg': None,
            'index_mcp_spread_deg': None,
            'index_mcp_flex_deg': None,
            'index_pip_flex_deg': None,
            'index_dip_flex_deg': None,
            'middle_mcp_flex_deg': None,
            'middle_pip_flex_deg': None,
            'ring_mcp_flex_deg': None,
            'ring_pip_flex_deg': None,
            'pinky_mcp_flex_deg': None,
            'pinky_pip_flex_deg': None,
            'thumb_cmc1_deg': None,
            'thumb_cmc2_deg': None,
            'thumb_mcp_deg': None,
            'thumb_ip_deg': None,
            'connected': False,
            'timestamp': None,
            'use_rgb_source': True,
            'hand_confidence': 0.0 
        }
        
        # Debug tracking
        self.last_confidence = None
        self.confidence_stuck_count = 0
        
        self.intel_data = {
            'yaw_deg': None,
            'pitch_deg': None,
            'roll_deg': None,
            'wrist_camera_x': None,
            'wrist_camera_y': None,
            'wrist_depth_z': None,
            'wrist_world_x': None,
            'wrist_world_y': None,
            'wrist_world_z': None,
            'depth_available': None,
            'index_mcp_spread_deg': None,
            'index_mcp_flex_deg': None,
            'index_pip_flex_deg': None,
            'index_dip_flex_deg': None,
            'middle_mcp_flex_deg': None,
            'middle_pip_flex_deg': None,
            'ring_mcp_flex_deg': None,
            'ring_pip_flex_deg': None,
            'pinky_mcp_flex_deg': None,
            'pinky_pip_flex_deg': None,
            'thumb_cmc1_deg': None,
            'thumb_cmc2_deg': None,
            'thumb_mcp_deg': None,
            'thumb_ip_deg': None,
            'connected': False,
            'timestamp': None,
            'use_rgb_source': False,
            'confidence': 0.0
        }
        
        # Combined data - this is what gets used by the controller
        self.combined_data = {
            'joint_positions': None,
            'wrist_position': None,
            'wrist_orientation': None,
            'adapt_qpos': None,
            'yaw_deg': None,
            'pitch_deg': None,
            'roll_deg': None,
            'index_mcp_spread_deg': None,
            'index_mcp_flex_deg': None,
            'index_pip_flex_deg': None,
            'index_dip_flex_deg': None,
            'middle_mcp_flex_deg': None,
            'middle_pip_flex_deg': None,
            'ring_mcp_flex_deg': None,
            'ring_pip_flex_deg': None,
            'pinky_mcp_flex_deg': None,
            'pinky_pip_flex_deg': None,
            'thumb_cmc1_deg': None,
            'thumb_cmc2_deg': None,
            'thumb_mcp_deg': None,
            'thumb_ip_deg': None,
            'connected': False,
            'timestamp': None,
            'active_source': 'rgb',  # 'rgb' or 'intel'
            'use_rgb_source': True,  # Control flag from RGB source
            'confidence': 0.8,       # Default confidence for dual source
            'source': 'dual'         # Source type identifier
        }
        
        self.rgb_lock = threading.Lock()
        self.intel_lock = threading.Lock()
        self.combined_lock = threading.Lock()
        
        # Server configurations
        self.rgb_host = 'localhost'
        self.rgb_port = 8888    # RGB source port
        self.intel_host = 'localhost' 
        self.intel_port = 8889  # Intel source port
        
        logger.info("Dual Source Data Receiver initialized")
        logger.info(f"RGB Source: {self.rgb_host}:{self.rgb_port}")
        logger.info(f"Intel Source: {self.intel_host}:{self.intel_port}")

    def start_clients(self):
        """Start TCP clients for both RGB and Intel sources"""
        # Start RGB source client
        rgb_thread = threading.Thread(target=self._rgb_client_thread, daemon=True)
        rgb_thread.start()
        
        # Start Intel source client
        intel_thread = threading.Thread(target=self._intel_client_thread, daemon=True)
        intel_thread.start()

    def _rgb_client_thread(self):
        """Client thread for RGB source (port 8888)"""
        while True:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(5.0)
                client_socket.connect((self.rgb_host, self.rgb_port))
                
                with self.rgb_lock:
                    self.rgb_data['connected'] = True
                
                while True:
                    try:
                        client_socket.settimeout(1.0)
                        data = client_socket.recv(4096)
                        if not data:
                            break
                            
                        # Handle multiple JSON messages
                        messages = data.decode().strip().split('\n')
                        for message in messages:
                            if message.strip():
                                try:
                                    json_data = json.loads(message)
                                    if json_data.get('type') != 'heartbeat':
                                        self._process_rgb_data(json_data)
                                except json.JSONDecodeError as e:
                                    continue
                                    
                    except socket.timeout:
                        continue
                    except Exception as e:
                        break
                        
            except Exception as e:
                with self.rgb_lock:
                    self.rgb_data['connected'] = False
                time.sleep(5)  

    def _intel_client_thread(self):
        # Client thread for Intel source (port 8889)
        while True:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(5.0)
                client_socket.connect((self.intel_host, self.intel_port))
                
                with self.intel_lock:
                    self.intel_data['connected'] = True
                
                # Send heartbeat to Intel source
                heartbeat = json.dumps({"type": "heartbeat"}) + '\n'
                client_socket.send(heartbeat.encode())
                
                last_heartbeat = time.time()
                
                while True:
                    try:
                        # Send periodic heartbeat
                        current_time = time.time()
                        if current_time - last_heartbeat > 10:  # Every 10 seconds
                            heartbeat = json.dumps({"type": "heartbeat"}) + '\n'
                            client_socket.send(heartbeat.encode())
                            last_heartbeat = current_time
                        
                        client_socket.settimeout(1.0)
                        data = client_socket.recv(4096)
                        if not data:
                            break
                            
                        # Handle multiple JSON messages
                        messages = data.decode().strip().split('\n')
                        for message in messages:
                            if message.strip():
                                try:
                                    json_data = json.loads(message)
                                    if json_data.get('type') != 'heartbeat':
                                        self._process_intel_data(json_data)
                                except json.JSONDecodeError:
                                    continue
                                    
                    except socket.timeout:
                        continue
                    except Exception as e:
                        break
                        
            except Exception as e:
                with self.intel_lock:
                    self.intel_data['connected'] = False
                time.sleep(5)

    def _process_rgb_data(self, data):
        # Process data from RGB source
        with self.rgb_lock:
            for key in self.rgb_data.keys():
                if key in data:
                    old_value = self.rgb_data[key]
                    new_value = data[key]
                    self.rgb_data[key] = new_value
                        
            self.rgb_data['timestamp'] = data.get('timestamp', time.time())
        
        self._update_combined_data()

    def _process_intel_data(self, data):
        # Process data from Intel source
        with self.intel_lock:
            for key in self.intel_data.keys():
                if key in data:
                    self.intel_data[key] = data[key]
            self.intel_data['timestamp'] = data.get('timestamp', time.time())
        
        self._update_combined_data()

    def _update_combined_data(self):
        # Update combined data based on comparing confidence from both RGB and Intel sources
        with self.combined_lock:
            # Get confidence values from both sources
            rgb_confidence = self.rgb_data.get('hand_confidence', 0.0)
            intel_confidence = self.intel_data.get('confidence', 0.0)
            
            rgb_timestamp = self.rgb_data.get('timestamp', 0)
            intel_timestamp = self.intel_data.get('timestamp', 0)
            current_time = time.time()
            
            # Check if data is stale (older than 5 seconds) and reduce confidence accordingly
            rgb_data_age = current_time - rgb_timestamp if rgb_timestamp else float('inf')
            intel_data_age = current_time - intel_timestamp if intel_timestamp else float('inf')
            
            if rgb_data_age > 5.0:
                rgb_confidence = 0.0
            if intel_data_age > 5.0:
                intel_confidence = 0.0
            
            # Compare confidence values and use the source with higher confidence
            if rgb_confidence >= intel_confidence:
                # Use RGB source data (higher or equal confidence)
                active_source = 'rgb'
                source_data = self.rgb_data.copy()
                print(f"Using RGB source data (RGB confidence: {rgb_confidence:.3f} >= Intel confidence: {intel_confidence:.3f})")
            else:
                # Use Intel source data (higher confidence)
                active_source = 'intel'
                source_data = self.intel_data.copy()
                print(f"Using Intel source data (Intel confidence: {intel_confidence:.3f} > RGB confidence: {rgb_confidence:.3f})")
                
            # Update combined data
            for key in self.combined_data.keys():
                if key in source_data:
                    self.combined_data[key] = source_data[key]
                    
            self.combined_data['active_source'] = active_source
            self.combined_data['connected'] = True  # Always connected to one source
            
            # Calculate confidence based on data quality
            confidence = 0.5  # Base confidence
            
            # Increase confidence if we have finger data
            if source_data.get('index_mcp_flex_deg') is not None:
                confidence += 0.2
            if source_data.get('yaw_deg') is not None and source_data.get('pitch_deg') is not None:
                confidence += 0.2
            if source_data.get('thumb_cmc1_deg') is not None:
                confidence += 0.1
                
            self.combined_data['confidence'] = min(confidence, 1.0)
            self.combined_data['source'] = f'dual-{active_source}'
            
            global received_retargeting_data
            with data_lock:
                for key in received_retargeting_data.keys():
                    if key in self.combined_data:
                        received_retargeting_data[key] = self.combined_data[key]
                received_retargeting_data['connected'] = self.combined_data['connected']

    def get_retargeting_data(self):
        # Get current retargeting data from active source
        with self.combined_lock:
            return self.combined_data.copy()

    def print_status(self):
        # Print connection status and active source based on RGB confidence
        rgb_status = "Connected" if self.rgb_data['connected'] else "Disconnected"
        intel_status = "Connected" if self.intel_data['connected'] else "Disconnected"
        active = self.combined_data.get('active_source', 'none')
        rgb_confidence = self.rgb_data.get('hand_confidence', 0.0)

    def run(self):
        self.start_clients()


# Global variables for smoothing wrist movements (for RGB retargeting controller)
previous_rgb_pitch = 0.0
previous_rgb_roll = 0.0
rgb_smoothing_factor = 0.8  # Higher = more smoothing, lower = more responsive

# Add global variables for finger joint smoothing (for RGB retargeting controller)
previous_rgb_finger_joints = {
    "Index_MCP": 0.0,
    "Index_PIP": 0.0,
    "Middle_MCP": 0.0,
    "Middle_PIP": 0.0,
    "Ring_MCP": 0.0,
    "Ring_PIP": 0.0,
    "Pinky_MCP": 0.0,
    "Pinky_PIP": 0.0,
    "Thumb_MCP": 0.0,
    "Thumb_CMC1": 0.0,
    "Thumb_CMC2": 0.0,
}

# Smoothing factors for different joint types (RGB retargeting controller)
rgb_wrist_smoothing = 0.8
rgb_roll_smoothing = 0.012
rgb_finger_smoothing = 0.7
rgb_thumb_smoothing = 0.85

# Maximum change per frame (safety limits) (RGB retargeting controller)
rgb_max_joint_change_per_frame = {
    "wrist": 0.05,    # 0.05 rad per frame max change
    "roll": 0.5,      # Reduced max change for roll
    "finger": 0.08,   # 0.08 rad per frame max change  
    "thumb": 0.03,    # 0.03 rad per frame max change
}

def apply_rgb_joint_smoothing_and_safety(new_joints, joint_type="finger"):
    #Apply smoothing and safety limits to joint positions for RGB retargeting controller.
    

    global previous_rgb_finger_joints
    
    smoothed_joints = {}
    
    # Select smoothing parameters based on joint type
    if joint_type == "wrist":
        smoothing = rgb_wrist_smoothing
        max_change = rgb_max_joint_change_per_frame["wrist"]
    elif joint_type == "thumb":
        smoothing = rgb_thumb_smoothing
        max_change = rgb_max_joint_change_per_frame["thumb"]
    else: 
        smoothing = rgb_finger_smoothing
        max_change = rgb_max_joint_change_per_frame["finger"]
    
    for joint_name, new_value in new_joints.items():
        # Get previous value (default to current if not tracked)
        prev_value = previous_rgb_finger_joints.get(joint_name, new_value)
        
        smoothed_value = smoothing * prev_value + (1 - smoothing) * new_value
        
        change = smoothed_value - prev_value
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
            smoothed_value = prev_value + change
        
        if "MCP" in joint_name or "PIP" in joint_name:
            smoothed_value = np.clip(smoothed_value, -1.8, 0.2)  
        elif "Yaw" in joint_name or "Pitch" in joint_name:
            smoothed_value = np.clip(smoothed_value, -0.6, 0.6)
        elif "CMC" in joint_name:
            smoothed_value = np.clip(smoothed_value, -1.0, 1.2)
        elif "IP" in joint_name:
            smoothed_value = np.clip(smoothed_value, -2.0, 0.2)
        
        smoothed_joints[joint_name] = smoothed_value
        
        # Update previous value for next frame
        previous_rgb_finger_joints[joint_name] = smoothed_value
    
    return smoothed_joints

def map_rgb_finger_angles_to_isaac_joints(mcp_spread_deg, mcp_flex_deg, pip_flex_deg, dip_flex_deg):
    """    
    Updated ranges based on testing:
    - MCP Spread: FIXED at straight position (1.500 rad)
    - MCP Flex range: +120° to +165° (where +165° = straight, +120° = fully closed) → 0 to -1.57 rad
    - PIP Flex range: +120° to +165° (where +165° = straight, +120° = fully closed) → 0 to -1.15 rad
    - DIP: FIXED at straight position (-1.044 rad)
    """
    mapped_joints = {}
    
    # Fix MCP spread at straight position
    mapped_joints["Index_MCP_Spread"] = 1.500  # Fixed at straight position
    
    if mcp_flex_deg is not None:
        # Map MCP flexion: +120° to +165° range (165° = straight, 120° = closed)
        flex_normalized = (165.0 - mcp_flex_deg) / (165.0 - 120.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        mcp_joint_value = flex_normalized * (-1.57)  # 0 to -1.57 rad
        mapped_joints["Index_MCP"] = np.clip(mcp_joint_value, -1.57, 0.0)
    
    if pip_flex_deg is not None:
        # Map PIP flexion: +120° to +165° range (165° = straight, 120° = fully closed)
        flex_normalized = (165.0 - pip_flex_deg) / (165.0 - 120.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        pip_joint_value = flex_normalized * (-1.15)  # 0 to -1.15 rad
        mapped_joints["Index_PIP"] = np.clip(pip_joint_value, -1.15, 0.0)
    
    # Fix DIP at straight position
    mapped_joints["Index_DIP"] = -1.044  # Fixed at straight position
    
    # Apply smoothing and safety checks
    smoothed_joints = apply_rgb_joint_smoothing_and_safety(mapped_joints, "finger")
    
    return smoothed_joints

def map_rgb_middle_finger_angles_to_isaac_joints(mcp_flex_deg, pip_flex_deg):
    """Map RGB retargeting middle finger angles to Isaac Lab joint positions."""
    mapped_joints = {}
    
    if mcp_flex_deg is not None:
        # Map MCP flexion: 150° (straight) to 110° (closed)
        flex_normalized = (150.0 - mcp_flex_deg) / (150.0 - 110.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        mcp_joint_value = flex_normalized * (-1.57)  # 0 to -1.57 rad
        mapped_joints["Middle_MCP"] = np.clip(mcp_joint_value, -1.57, 0.0)
    
    if pip_flex_deg is not None:
        # Map PIP flexion: 175° (straight) to 100° (closed)
        flex_normalized = (175.0 - pip_flex_deg) / (175.0 - 100.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        pip_joint_value = flex_normalized * (-1.15)  # 0 to -1.15 rad
        mapped_joints["Middle_PIP"] = np.clip(pip_joint_value, -1.15, 0.0)
    
    # Apply smoothing and safety checks
    smoothed_joints = apply_rgb_joint_smoothing_and_safety(mapped_joints, "finger")
    
    return smoothed_joints

def map_rgb_ring_finger_angles_to_isaac_joints(mcp_flex_deg, pip_flex_deg):
    """Map RGB retargeting ring finger angles to Isaac Lab joint positions."""
    mapped_joints = {}
    
    if mcp_flex_deg is not None:
        # Map MCP flexion: 130° (straight) to 110° (closed)
        flex_normalized = (130.0 - mcp_flex_deg) / (130.0 - 110.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        mcp_joint_value = flex_normalized * (-1.57)  # 0 to -1.57 rad
        mapped_joints["Ring_MCP"] = np.clip(mcp_joint_value, -1.57, 0.0)
    
    if pip_flex_deg is not None:
        # Map PIP flexion: 170° (straight) to 100° (closed)
        flex_normalized = (170.0 - pip_flex_deg) / (170.0 - 100.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        pip_joint_value = flex_normalized * (-1.15)  # 0 to -1.15 rad
        mapped_joints["Ring_PIP"] = np.clip(pip_joint_value, -1.15, 0.0)

    # Apply smoothing and safety checks
    smoothed_joints = apply_rgb_joint_smoothing_and_safety(mapped_joints, "finger")
    
    return smoothed_joints

def map_rgb_pinky_finger_angles_to_isaac_joints(mcp_flex_deg, pip_flex_deg):
    """Map RGB retargeting pinky finger angles to Isaac Lab joint positions."""
    mapped_joints = {}
    
    if mcp_flex_deg is not None:
        # Map MCP flexion: 140° (straight) to 100° (closed)
        flex_normalized = (140.0 - mcp_flex_deg) / (140.0 - 100.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        mcp_joint_value = flex_normalized * (-1.57)  # 0 to -1.57 rad
        mapped_joints["Pinky_MCP"] = np.clip(mcp_joint_value, -1.57, 0.0)
    
    if pip_flex_deg is not None:
        # Map PIP flexion: 170° (straight) to 110° (closed)
        flex_normalized = (170.0 - pip_flex_deg) / (170.0 - 110.0)  # 0 = straight, 1 = closed
        flex_normalized = np.clip(flex_normalized, 0.0, 1.0)
        pip_joint_value = flex_normalized * (-1.15)  # 0 to -1.15 rad
        mapped_joints["Pinky_PIP"] = np.clip(pip_joint_value, -1.15, 0.0)
    
    # Apply smoothing and safety checks
    smoothed_joints = apply_rgb_joint_smoothing_and_safety(mapped_joints, "finger")
    
    return smoothed_joints

def map_rgb_thumb_angles_to_isaac_joints(cmc1_deg, cmc2_deg, mcp_deg, ip_deg):
    """Map RGB retargeting thumb angles to Isaac Lab joint positions."""
    mapped_joints = {}
    
    if cmc1_deg is not None:
        # Map CMC1: 45° (straight) = 1.0, 37° (closed) = 0.0
        cmc1_clamped = max(37.0, min(45.0, cmc1_deg))
        cmc1_normalized = (cmc1_clamped - 37.0) / (45.0 - 37.0)  # 0 = closed, 1 = straight
        mapped_joints["Thumb_CMC1"] = cmc1_normalized  # 0.0 to 1.0 range
    
    if cmc2_deg is not None:
        # Map CMC2: 170° (left) = 0.0, 150° (right) = -0.7
        cmc2_clamped = max(150.0, min(170.0, cmc2_deg))
        cmc2_normalized = (170.0 - cmc2_clamped) / (170.0 - 150.0)  # 0 = left, 1 = right
        cmc2_joint_value = cmc2_normalized * (-0.7)  # 0.0 to -0.7 range
        mapped_joints["Thumb_CMC2"] = np.clip(cmc2_joint_value, -0.7, 0.0)
    
    if ip_deg is not None:
        # Map MCP using IP angles: 170° (straight) = 0.0, 130° (closed) = -1.9
        ip_clamped = max(130.0, min(170.0, ip_deg))
        ip_normalized = (170.0 - ip_clamped) / (170.0 - 130.0)  # 0 = straight, 1 = closed
        mcp_joint_value = ip_normalized * (-1.9)  # 0.0 to -1.9 range
        mapped_joints["Thumb_MCP"] = np.clip(mcp_joint_value, -1.9, 0.0)
    
    # Apply smoothing and safety checks with higher smoothing for thumb
    smoothed_joints = apply_rgb_joint_smoothing_and_safety(mapped_joints, "thumb")
    
    return smoothed_joints

def map_rgb_retargeting_to_isaac_joints(yaw_deg=140.0, pitch_deg=-70.0, roll_deg=0.0):
    """
    Map RGB retargeting yaw/pitch/roll values to Isaac Lab joint positions for wrist orientation.
    
    User specifications:
    - Straight hand position: yaw = +140°, pitch = -70°
    - Yaw range: +115° (left) to +165° (right)  
    - Pitch range: -90° (up) to -20° (down)
    - Roll mapping with custom range
    """
    global previous_rgb_pitch, previous_rgb_roll
    
    mapped_joints = {}
    
    # Map yaw degrees to joint radian values
    yaw_clamped = max(115.0, min(165.0, yaw_deg))
    yaw_normalized = (yaw_clamped - 140.0) / 50.0  # -1.0 to +1.0 range around resting
    wrist_yaw_joint = yaw_normalized * 0.5  # Scale to ±0.5 radians
    
    # Map pitch degrees to joint radian values  
    pitch_clamped = max(-90.0, min(-20.0, pitch_deg))
    pitch_normalized = (pitch_clamped - (-70.0)) / 70.0  # -1.0 to +1.0 range around resting
    wrist_pitch_joint = pitch_normalized * 0.5  # Scale to ±0.5 radians
    
    # Map roll degrees to wrist_3_joint with INVERTED mapping
    roll_clamped = max(-180.0, min(180.0, roll_deg))
    wrist_roll_joint = -np.radians(roll_clamped)  # INVERTED: Negative sign added
    
    # Apply safety limits
    wrist_yaw_joint = np.clip(wrist_yaw_joint, -0.6, 0.6)
    wrist_pitch_joint = np.clip(wrist_pitch_joint, -0.6, 0.6)
    wrist_roll_joint = np.clip(wrist_roll_joint, -np.pi, np.pi)  # Full rotation allowed
    
    # Apply different smoothing for pitch and roll
    wrist_pitch_joint = rgb_wrist_smoothing * previous_rgb_pitch + (1 - rgb_wrist_smoothing) * wrist_pitch_joint
    wrist_roll_joint = rgb_roll_smoothing * previous_rgb_roll + (1 - rgb_roll_smoothing) * wrist_roll_joint
    
    # Apply maximum change limits with different limits for roll
    pitch_change = wrist_pitch_joint - previous_rgb_pitch
    if abs(pitch_change) > rgb_max_joint_change_per_frame["wrist"]:
        pitch_change = rgb_max_joint_change_per_frame["wrist"] if pitch_change > 0 else -rgb_max_joint_change_per_frame["wrist"]
        wrist_pitch_joint = previous_rgb_pitch + pitch_change
    
    roll_change = wrist_roll_joint - previous_rgb_roll
    if abs(roll_change) > rgb_max_joint_change_per_frame["roll"]:
        roll_change = rgb_max_joint_change_per_frame["roll"] if roll_change > 0 else -rgb_max_joint_change_per_frame["roll"]
        wrist_roll_joint = previous_rgb_roll + roll_change
    
    # Update previous values
    previous_rgb_pitch = wrist_pitch_joint
    previous_rgb_roll = wrist_roll_joint

    # Return wrist orientation joints including roll via wrist_3_joint
    wrist_joints = {
        "Wrist_Yaw": wrist_yaw_joint,
        "Wrist_Pitch": wrist_pitch_joint,
        "wrist_3_joint": wrist_roll_joint  
    }
    
    # Apply smoothing and safety checks for wrist joints (but skip roll since we handled it above)
    smoothed_wrist_joints = apply_rgb_joint_smoothing_and_safety({k: v for k, v in wrist_joints.items() if k != "wrist_3_joint"}, "wrist")
    smoothed_wrist_joints["wrist_3_joint"] = wrist_roll_joint  
    
    return smoothed_wrist_joints

def main():
    # Main function that runs the Isaac Lab simulation with retargeting control.
    global received_retargeting_data

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    robot = design_scene()
    sim.reset()
    receiver = DualSourceDataReceiver()
    receiver.start_clients()
    sim_dt = sim.get_physics_dt()
    count = 0
    
    joint_names = robot.joint_names
    print(f"[INFO]: Found {len(joint_names)} joints")
    
    # Check which thumb joints are available in the robot
    thumb_joints_in_robot = [j for j in joint_names if 'Thumb' in j]
    print(f"[DEBUG]: Available thumb joints in robot: {thumb_joints_in_robot}")
    
    # Check which hand orientation joints are available
    orientation_joints_in_robot = [j for j in joint_names if j in ['Wrist_Yaw', 'Wrist_Pitch', 'Wrist_Roll']]
    print(f"[DEBUG]: Available hand orientation joints in robot: {orientation_joints_in_robot}")
    
    # Check for arm roll joint
    arm_roll_joint = "wrist_3_joint"
    has_arm_roll = arm_roll_joint in joint_names
    print(f"[DEBUG]: Arm roll joint '{arm_roll_joint}' available: {has_arm_roll}")
    
    # Fixed positions for ALL joints except controlled finger and orientation joints
    fixed_positions = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.2,
        "elbow_joint": 1.8,
        "wrist_1_joint": -0.6,
        "wrist_2_joint": 0.0,
        # Index finger joints - spread and DIP fixed, MCP and PIP controlled
        "Index_MCP_Spread": -0.1, 
        "Index_DIP": -1.0,        
        # Other finger DIP joints fixed
        "Middle_DIP": -1.0,       
        "Ring_DIP": -1.0,          
        "Pinky_DIP": -1.0,         
    }
    
    print("[INFO]: Setup complete...")
    print("[INFO]: DUAL SOURCE CONTROLLER SYSTEM")
    print("[INFO]: RGB Source on port 8888")
    print("[INFO]: Intel Source on port 8889")
    print("[INFO]: Automatic source switching based on RGB control flag")
    print("[INFO]: Robot in resting position - ready for dual source control")

    # Simulate physics
    while simulation_app.is_running():
        retargeting_data = receiver.get_retargeting_data()
        
        # Create joint position targets starting with current positions
        joint_pos_target = robot.data.joint_pos.clone()
        
        # Apply fixed positions for non-controlled joints
        for i, joint_name in enumerate(joint_names):
            if joint_name in fixed_positions:
                joint_pos_target[0, i] = fixed_positions[joint_name]
        
        # Apply retargeting control if data is available
        if retargeting_data:
            confidence = retargeting_data.get('confidence', 0.8)
            source = retargeting_data.get('source', 'dual')
            
            # Get current data from the active source
            current_data = retargeting_data.copy()
            
            # Check if we have valid finger data
            has_finger_data = (current_data.get('index_mcp_flex_deg') is not None and
                              current_data.get('index_pip_flex_deg') is not None and
                              current_data.get('middle_mcp_flex_deg') is not None and
                              current_data.get('middle_pip_flex_deg') is not None and
                              current_data.get('ring_mcp_flex_deg') is not None and
                              current_data.get('ring_pip_flex_deg') is not None and
                              current_data.get('pinky_mcp_flex_deg') is not None and
                              current_data.get('pinky_pip_flex_deg') is not None)
            
            has_thumb_data = (current_data.get('thumb_cmc1_deg') is not None and
                             current_data.get('thumb_cmc2_deg') is not None and
                             current_data.get('thumb_ip_deg') is not None)
            
            has_wrist_data = (current_data.get('yaw_deg') is not None and
                             current_data.get('pitch_deg') is not None and
                             current_data.get('roll_deg') is not None)
            
            if has_finger_data:
                finger_joints = {}
                
                index_joints = map_rgb_finger_angles_to_isaac_joints(
                    current_data.get('index_mcp_spread_deg'),
                    current_data.get('index_mcp_flex_deg'),
                    current_data.get('index_pip_flex_deg'),
                    current_data.get('index_dip_flex_deg')
                )
                finger_joints.update(index_joints)
                
                middle_joints = map_rgb_middle_finger_angles_to_isaac_joints(
                    current_data.get('middle_mcp_flex_deg'),
                    current_data.get('middle_pip_flex_deg')
                )
                finger_joints.update(middle_joints)
                
                ring_joints = map_rgb_ring_finger_angles_to_isaac_joints(
                    current_data.get('ring_mcp_flex_deg'),
                    current_data.get('ring_pip_flex_deg')
                )
                finger_joints.update(ring_joints)
                
                pinky_joints = map_rgb_pinky_finger_angles_to_isaac_joints(
                    current_data.get('pinky_mcp_flex_deg'),
                    current_data.get('pinky_pip_flex_deg')
                )
                finger_joints.update(pinky_joints)
                
                for i, joint_name in enumerate(joint_names):
                    if joint_name in finger_joints:
                        joint_pos_target[0, i] = finger_joints[joint_name]
            
            if has_thumb_data:
                thumb_joints = map_rgb_thumb_angles_to_isaac_joints(
                    current_data.get('thumb_cmc1_deg'),
                    current_data.get('thumb_cmc2_deg'),
                    current_data.get('thumb_mcp_deg'),
                    current_data.get('thumb_ip_deg')
                )
                
                for i, joint_name in enumerate(joint_names):
                    if joint_name in thumb_joints:
                        joint_pos_target[0, i] = thumb_joints[joint_name]
            
            if has_wrist_data:
                wrist_joints = map_rgb_retargeting_to_isaac_joints(
                    current_data.get('yaw_deg', 140.0),
                    current_data.get('pitch_deg', -70.0),
                    current_data.get('roll_deg', 0.0)
                )
                
                for i, joint_name in enumerate(joint_names):
                    if joint_name in wrist_joints:
                        joint_pos_target[0, i] = wrist_joints[joint_name]

        
        # Validate joint_pos_target before setting
        has_nan = torch.isnan(joint_pos_target).any()
        has_inf = torch.isinf(joint_pos_target).any()
        
        if has_nan or has_inf:
            print(f"[WARNING]: Invalid joint targets detected at step {count} - using current positions")
            joint_pos_target = robot.data.joint_pos.clone()
        
        # Set joint position targets
        robot.set_joint_position_target(joint_pos_target)
        
        # Print control commands being sent to Isaac
        if count % 30 == 0:  # Print every 30 frames 
            active_commands = []
            for i, joint_name in enumerate(joint_names):
                if joint_pos_target[0, i] != robot.data.joint_pos[0, i]:  # If position changed
                    target_value = joint_pos_target[0, i].item()
                    active_commands.append(f"{joint_name}={target_value:+6.3f}")
        
        robot.write_data_to_sim()
        
        sim.step()
        
        robot.update(sim_dt)
        
        count += 1

if __name__ == "__main__":
    
    main()
    simulation_app.close()
