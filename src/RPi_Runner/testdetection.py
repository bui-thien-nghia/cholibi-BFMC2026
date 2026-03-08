from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

class Pi5Camera:
    def __init__(self, width=640, height=480):
        # Initialize the official Pi5 camera library
        self.picam2 = Picamera2()
        
        # Configure it for BGR video (OpenCV standard)
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        
        # Start the camera continuously
        self.picam2.start()

    def read(self):
        # Grab the latest frame directly as a numpy array
        try:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None
            return True, frame
        except Exception as e:
            print(f"Picamera2 Error: {e}")
            return False, None

    def release(self):
        self.picam2.stop()
        self.picam2.close()
        
def main():
    cam = Pi5Camera()
    model = YOLO('./obj_det/model/obj.onnx')
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        r = model.predict(
            frame,
            half=True,
            imgsz=320,
            conf=0.7,
            verbose=False
        )[0]
        img = r.plot()
        cv2.imshow('check', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cap.destroyAllWindows()
            return
        
if __name__ == '__main__':
    main()
