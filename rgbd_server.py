import cv2
import socket
import struct
import pickle
import numpy as np
import pyrealsense2 as rs
import time

print("Initializing RealSense camera...")
pipeline = rs.pipeline()
config = rs.config()

#color and depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Get camera intrinsics
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

camera_intrinsics = {
    'fx': depth_intrinsics.fx,
    'fy': depth_intrinsics.fy,
    'ppx': depth_intrinsics.ppx,
    'ppy': depth_intrinsics.ppy,
    'width': depth_intrinsics.width,
    'height': depth_intrinsics.height,
    'coeffs': depth_intrinsics.coeffs
}

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9998))
server_socket.listen(1)
conn, addr = server_socket.accept()

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        continue
    
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    data_package = {
        'color_frame': color_image,
        'depth_frame': depth_image,
        'intrinsics': camera_intrinsics,
        'timestamp': time.time()
    }
    
    data = pickle.dumps(data_package)
    # Send message length first (as 4 bytes, unsigned int)
    conn.sendall(struct.pack(">L", len(data)) + data)
    
    # Visual feedback can be removed for performance
    cv2.imshow('Intel RealSense Feed', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()   