#!/usr/bin/env python3

# This ROS node publishes the images captured by the webcam (or the default
# camera, if any) of the machine in which it is executed
 
# We import the necessary libraries, packages and types
import rospy # Python library for ROS
import cv2 # OpenCV library
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from sensor_msgs.msg import Image # Message type in which we will publish the images

# In this function, we will publish the images
def publish_message():
 
  # This node is going to publish messages of type 'Image' via the topic 
  # 'video_frames', with a limit of queued messages of 10
  pub = rospy.Publisher('video_frames', Image, queue_size=10)
  
  # We tell rospy the name of the node. Anonymous = True ensures the node has a 
  # unique name, by adding random numbers to the end of the name
  rospy.init_node('video_pub_py', anonymous=True)
  
  # We create a Rate object, giving it the argument 10, so that we go through 
  # the loop 10 times per second
  rate = rospy.Rate(10)
  
  # We create a VideoCapture object. The argument '0' gets the default webcam
  cap = cv2.VideoCapture(0)
  
  # The following line is needed to convert between ROS and OpenCV images
  br = CvBridge()
 
  # While ROS is still running...
  while not rospy.is_shutdown():
     
      # We capture the current image from the webcam and store it in 'frame'.
      # The method 'cap.read()' also returns True or False, depending on the
      # success of capturing an image
      ret, frame = cap.read()
      
      # If we have been able to capture an image...
      if ret == True:
             
        # We publish the image ('frame') via the topic 'video_frames'. But first,
        # we convert the OpenCV image to a ROS image message with the method
        # 'cv2_to_imgmsg'
        pub.publish(br.cv2_to_imgmsg(frame))
             
      # We sleep just long enough to maintain the desired rate through the loop
      rate.sleep()

# We publish messages until Ctrl-C is pressed
if __name__ == '__main__':
  try:
    publish_message()
  except rospy.ROSInterruptException:
    pass

