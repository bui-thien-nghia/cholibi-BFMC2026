from picamera2 import Picamera2

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
