#!/usr/bin/env python3
 
# We import the necessary libraries, packages and types
import rospy # Python library for ROS
from standard_msgs.msg import String # Message type in which the letters are received

from mobility import Mobility

# This function is executed every time an image is received 
def callback(data):
  rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def receive_message():
 
  # We tell rospy the name of the node. Anonymous = True ensures the node has a 
  # unique name, by adding random numbers to the end of the name
  rospy.init_node('mobiity_node_py', anonymous=True)
  
  # This node is subscribed to the topic 'webcam_sub', whose message type is string
  # and whose callback function is 'callback'
  rospy.Subscriber('letter', String, callback)
 
  # MOBILITY:
  
  bot = Mobility(20, 19, 21, 26, 16, 13)


  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()
 
# We call the function 'receive_message()' constantly
if __name__ == '__main__':
  receive_message()

