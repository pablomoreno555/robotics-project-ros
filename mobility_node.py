#!/usr/bin/env python3

# We import the necessary libraries, packages and types
import rospy # Python library for ROS
from standard_msgs.msg import String # Message type in which the letters are received

from mobility import Mobility

# This function is executed every time a command is received
# data is the String of the given ASL command
# bot is the mobility object
def callback(data, bot):
    # Logs the id, the command data, and the mobility object
    rospy.loginfo(f'{rospy.get_caller_id()}: The command was {data.data} executed by {bot}')

    # This is the command factory mapping the known commands to the mobility movements
    if data.data == 'f':
        bot.forward()

    elif data.data == 'b':
        bot.backward()

    elif data.data == 'r':
        bot.go_right()

    elif data.data == 'l':
        bot.go_left()

    elif data.data == 'c':
        bot.square()

    elif data.data == 's':
        bot.spin()

    else:
        rospy.loginfo(f'Does not recognize the command {data.data}')

def receive_message(bot_obj):
    # Initializes the subscriber node with a base name
    # anonymous = True creates a unique name by adding random numerical suffix
    rospy.init_node('mobility_node', anonymous=True)

    # Creates a subscriber object to the topic 'letter'
    # String is the message type
    # callback is the function called when a message is recieved
    # bot_obj is the Mobility object passed into the callback function
    rospy.Subscriber('letter', String, callback, bot_obj)

    # Keeps the node running until ros is stopped
    rospy.spin()

# We call the function 'receive_message()' constantly
if __name__ == '__main__':
    # Creates the mobility object with designated pins
    bot = Mobility(20, 19, 21, 26, 16, 13)
    receive_message(bot)
