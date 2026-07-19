from ultralytics import YOLO
import cv2
import winsound

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, classes=[0,15,16,17,18,19])

    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    count = len(boxes)

    cv2.putText(
        annotated_frame,
        f"Count: {count}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    
    person_detected = False

    for box in boxes:
        cls = int(box.cls[0])

        if cls == 0:
            person_detected = True

    if person_detected:
        print("🚨 Person Detected!")
        winsound.Beep(1000, 500)

    cv2.imshow("Human & Animal Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()