from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")
img = cv2.imread("img1.jpg")
results = model(img, classes=[0,15,16,17,18,19])
annotated = results[0].plot()
cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()