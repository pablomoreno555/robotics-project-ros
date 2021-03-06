#!/usr/bin/env python3

# This ROS node is subscribed to the topic 'video_frames'. It receives the images
# captured by the webcam and displays them in a window created with OpenCV

# We import the necessary libraries, packages and types
import rospy  # Python library for ROS
import cv2  # OpenCV library
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
from sensor_msgs.msg import Image  # Message type in which the images are received
from std_msgs.msg import String
from tensorflow.keras.models import load_model
import numpy as np


class Recognizer:
    def __init__(self):
        self.pub = rospy.Publisher("letter", String, queue_size=5)
        self.model = load_model("/data/ASLModel.h5")
        self.classes = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
        ]

    # This function is executed every time an image is received
    def callback(self, data):

        # We convert the ROS image message received into an OpenCV image
        br = CvBridge()
        current_frame = br.imgmsg_to_cv2(data)

        img = cv2.resize(current_frame, (640, 480))
        img = np.reshape(img, (-1, 640, 480, 3))

        pred = self.model.predict(img)  # Softmax predictions
        pred = np.argmax(pred)  # Index of most confident prediction
        pred = self.classes[pred]  # Letter corresponding to index

        self.pub.publish(pred)

    def receive_message(self,):

        # We tell rospy the name of the node. Anonymous = True ensures the node has a
        # unique name, by adding random numbers to the end of the name
        rospy.init_node("video_sub_py", anonymous=True)

        # This node is subscribed to the topic 'video_frames', whose message type is
        # 'Image', and whose callback function is 'callback'
        rospy.Subscriber("video_frames", Image, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

        # When the node stops its execution, we close the 'camera' window
        cv2.destroyAllWindows()


# We call the function 'receive_message()' constantly
if __name__ == "__main__":
    rec = Recognizer()
    rec.receive_message()

