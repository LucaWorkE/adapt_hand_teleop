# Real-Time Hand Tracking & Robotic Control Pipeline

A comprehensive system for real-time hand tracking using dual camera sources (RGB and Intel RealSense RGBD) with robotic hand control integration. The pipeline supports MediaPipe-based hand detection, confidence-based source switching.

## System Architecture

### Core Components

**Camera Servers:**
- `rgb_server.py` - Standard RGB camera server (port 9999)
- `rgbd_server.py` - Intel RealSense RGBD camera server with depth data (port 9998)

**Hand Tracking & Retargeting:**
- `rgb_retargetting.py` - RGB-based hand tracking with MediaPipe
- `rgbd_retargetting.py` - RGBD-based hand tracking with depth-enhanced world coordinates
- `command_sender.py` - Dual-source data receiver with confidence-based switching
- `isaac_controller.py` - Isaac Lab simulation controller for virtual robot testing

## Start

### 1. Launch Camera Servers

**For RGB Camera:**
```bash
python rgb_server.py
```

**For Intel RealSense Camera:**
```bash
python rgbd_server.py
```

### 2. Start Hand Tracking Systems

**RGB Hand Tracking:**
```bash
python rgb_retargetting.py
```

**RGBD Hand Tracking:**
```bash
python rgbd_retargetting.py
```

### 3. Launch Robot Control

**For Physical Robot (via ROS):**
```bash
python command_sender.py
```

**For Isaac Lab Simulation:**
```bash
python isaac_controller.py
```
**Note that usd_path needs to be updated to the path where the ur5_adapt_hand.usd file is located**

## Data Flow

```
RGB Camera (9999) → rgb_retargetting.py (8890) ↘
                                                  → command_sender.py → ROS Robot
RGBD Camera (9998) → rgbd_retargetting.py (8893) ↗

RGB Camera (9999) → rgb_retargetting.py (8890) ↘
                                                  → Isaac lab simulation
RGBD Camera (9998) → rgbd_retargetting.py (8893) ↗
```

## Key Features

### Dual-Source Hand Tracking
- **RGB Source**: Standard camera with inferred depth using MediaPipe
- **RGBD Source**: Intel RealSense with precise depth data and world coordinates output
- **Confidence-Based Switching**: Automatically selects the most reliable source based on confidence variable value 

### Advanced Hand Analysis
- **21-Point Hand Landmarks**: Full MediaPipe hand detection
- **Joint Angle Calculation**: MCP, PIP, DIP angles for all fingers
- **Wrist Orientation**: Yaw, pitch, and roll estimation
- **World Coordinates**: 3D position mapping using camera intrinsics

### Robot Control Integration
- **Isaac Lab Support**: Virtual robot simulation and testing

### Calibration System
- **Interactive Calibration**: Keyboard-based yaw/pitch calibration
- **Range Mapping**: Custom joint ranges

## Configuration

### Hand Tracking Parameters
```python
# Detection confidence thresholds
min_detection_confidence = 0.8
min_tracking_confidence = 0.8
```

### Network Ports
- **RGB Camera Server**: 9999
- **RGBD Camera Server**: 9998
- **RGB Hand Data**: 8890
- **RGBD Hand Data**: 8893
- **Isaac Lab (RGB)**: 8888
- **Isaac Lab (RGBD)**: 8889
- **ROS Bridge**: 9090

## Controls

### Calibration (RGB Retargeting)
- **'t' Key**: Hold to calibrate yaw (left/right rotation)
- **'o' Key**: Hold to calibrate pitch (up/down rotation)
- **'q' Key**: Quit application

### Calibration (RGBD Retargeting)
- **'y' Key**: Hold to calibrate yaw (left/right rotation)
- **'p' Key**: Hold to calibrate pitch (up/down rotation)
- **'q' Key**: Quit application

## Dependencies

### Core Libraries
```bash
pip install mediapipe opencv-python numpy scipy 
pip install pyrealsense2 roslibpy loguru
pip install torch isaac-lab
pip install pynput
```

## Note that mediapipe works with pyhton 3.8 up to 3.10 as of september 2025
##

### Hardware Requirements
- **RGB Camera**: Any USB/network camera
- **Intel RealSense**: D435/D455 or compatible depth camera

## File Structure

```
camera_retargetting/
├── rgb_server.py              # RGB camera server
├── rgbd_server.py             # Intel RealSense server
├── rgb_retargetting.py        # RGB hand tracking
├── rgbd_retargetting.py       # RGBD hand tracking
├── command_sender.py          # Robot control bridge
├── isaac_controller.py        # Isaac Lab simulation
└── README.md                  # This file
```

## Technical Details

### Hand Landmark Processing
The system uses MediaPipe to detect 21 hand landmarks and converts them to MANO convention for robotic applications. Joint angles are computed using vector mathematics between landmark points.

### Depth Enhancement
RGBD tracking uses Intel RealSense depth data to provide accurate world coordinates.

### Confidence-Based Selection
The command sender automatically switches between RGB and RGBD sources based on detection confidence.
## Troubleshooting

### Common Issues
- **No camera feed**: Check camera connections and port availability
- **Poor tracking**: Adjust lighting conditions or camera position

## Performance

- **Tracking Rate**: 30 FPS (RGB/RGBD)
- **Latency**: <50ms end-to-end

## Acknowledgments

This project was developed with invaluable support from:

- **CREATE LAB EPFL** - Research infrastructure and technical guidance
- **ANYTELEOP** - Hand tracking methodologies and MANO conventions  
- **Supervision by JOSIE HUGHES AND CHENG PAN** - Project guidance and research direction
