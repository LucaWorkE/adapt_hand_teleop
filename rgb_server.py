import cv2
import socket
import struct
import pickle
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Camera properties 
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
            data = pickle.dumps(frame)
            # Send message length first (as 4 bytes, unsigned int)
            conn.sendall(struct.pack(">L", len(data)) + data)
            
            # Visual feedback can be removed for performance
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