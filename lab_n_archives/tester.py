import cv2
from ultralytics import YOLO

model_src = 'model/obj_tuned/weights/best.pt'
model = YOLO(model_src)
cap = cv2.VideoCapture('http://192.168.1.241:4747/video')
print('=====INITIATION COMPLETE=====')

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()