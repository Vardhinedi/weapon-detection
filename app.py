import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model
# For testing: use yolov8n.pt (COCO dataset, no weapons)
# For real weapon detection: replace with a weapon-trained model like "weapons-yolov8n.pt"
model = YOLO("yolov8n.pt")

# Weapon-related classes (adjust to match your trained model)
weapon_classes = ["knife", "gun", "pistol", "rifle"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    # Draw results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Show only weapons with confidence > 0.5
            if conf > 0.5 and label.lower() in weapon_classes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show video
    cv2.imshow("Weapon Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
