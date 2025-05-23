import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO("model\cricket_ball_detector.pt")

cap = cv2.VideoCapture("videos\\kohli.mp4")

# Open a file to write coordinates
coord_file = open("coordinates\coordinates.txt", "w")

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    black_frame = np.zeros_like(frame)

    results = model(frame, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()

        class_0_indices = [i for i, cls_id in enumerate(class_ids) if int(cls_id) == 0]

        if class_0_indices:
            best_idx = max(class_0_indices, key=lambda i: confidences[i])
            x1, y1, x2, y2 = map(int, boxes[best_idx])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Write to coordinates file
            coord_file.write(f"{frame_num},{cx},{cy}\n")
            print(f"Frame {frame_num}: Ball at ({cx},{cy})")

        else:
            print(f"Frame {frame_num}: No class 0 object detected")
    else:
        print(f"Frame {frame_num}: No object detected")

# Clean up
coord_file.close()
cap.release()
cv2.destroyAllWindows()
