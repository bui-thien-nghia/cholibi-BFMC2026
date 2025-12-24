import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from Perception_UNDER_DEVELOPMENT.lane_detect import run_lane_detect, add_lanes_to_image

class LaneDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.cv_image = np.zeros((2048, 1080))
        rospy.init_node('LANEnod', anonymous=True)

        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.process_frame)
        self.lane_pub = rospy.Publisher("/automobile/image_lane", Image, queue_size=1)


    def process_frame(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            polys = run_lane_detect(self.cv_image)
            processed_img = add_lanes_to_image(self.cv_image, polys)
            send_data = Image()
            send_data.data = processed_img
            self.lane_pub.publish(processed_img)
        except CvBridgeError as e:
            rospy.logerr(f"STATUS: CVBRIDGEERROR: {e}")
        except Exception as e:
            rospy.logerr(f"STATUS: ERROR: {e}")


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        nod = LaneDetectionNode()
        nod.run()
    except rospy.ROSInterruptException:
        pass