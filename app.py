import cv2
from ultralytics import YOLO

# Load the knife detection model
MODEL_PATH = r"C:\weapon-detection\Knife-Detector\Results\runs\detect\ADAM_LR_0_0005\weights\best.pt"
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Draw results directly with Ultralytics built-in plot
    annotated_frame = results[0].plot()

    # Show video with detections
    cv2.imshow("Knife Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
