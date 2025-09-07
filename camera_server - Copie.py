import cv2
import socket
import struct
import pickle
import time

# Camera setup with better configuration
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Socket setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)
print("Camera server listening on port 9999...")
print("Waiting for connection...")

try:
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            continue
            
        try:
            # Pickle the frame
            data = pickle.dumps(frame)
            # Send message length first (as 4 bytes, unsigned int)
            conn.sendall(struct.pack(">L", len(data)) + data)
            
            # Frame rate monitoring
            frame_count += 1
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Camera server FPS: {fps:.1f}, Frames sent: {frame_count}")
            
            # Visual feedback (optional - can be disabled for performance)
            cv2.imshow('Camera Server Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error sending frame: {e}")
            break
            
except KeyboardInterrupt:
    print("Shutting down camera server...")
except Exception as e:
    print(f"Server error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    server_socket.close()
    print("Camera server stopped")   