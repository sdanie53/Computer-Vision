import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam.

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening camera")

# Get the frame width and height
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f'Frame width: {frame_width}, Frame height: {frame_height}')

cap.release()
