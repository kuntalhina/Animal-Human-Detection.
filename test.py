from ultralytics import YOLO
import cv2
model = YOLO("yolo8n.pt")
img = cv2.imread("Horse.jpg")
results = model(img)
annotated = results[0].plot()
cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()