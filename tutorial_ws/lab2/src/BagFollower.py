#!/usr/bin/env python

import rospy, rosbag, rostopic
from ackermann_msgs.msg import AckermannDriveStamped

BAG_TOPIC = '/car/mux/ackermann_cmd_mux/input/teleop' # Name of the topic that should be extracted from the bag
PUB_TOPIC = '/car/mux/ackermann_cmd_mux/input/teleop'
# The rate at which messages should be published in simulation. In reality the joystick publishing rate increases over time, so it's not constant
PUB_RATE_SIM = 10 
INIT_PUB_RATE = 20 # Initial Publishing Rate for Real Car

# Loads a bag file, reads the msgs from the specified topic, and republishes them
def follow_bag(bag_path, follow_backwards=False):

	# Setup publisher that publishes to PUB_TOPIC
	pub = rospy.Publisher(PUB_TOPIC, AckermannDriveStamped, queue_size=10)

   	# Setup the out going AckermannDriveStamped message
   	car_trajectory = AckermannDriveStamped()
   	car_trajectory.header.frame_id = None
        
	with rosbag.Bag(bag_path) as bag:
		seq = 0
        # Get number of messages puplished to BAG_TOPIC
        # msg_cnt = bag.get_message_count(BAG_TOPIC)

		# bag.read_messages() returns tuple of tuples (topic, message, timestamp)
		# msg type AckermannDriveStamped (
		# Header: seq(uint32), stamp(time:two-int), frame_id; 
		# AckermannDrive: steering_angle(float32), steering_angle_velocity(float32),
		# speed(float32), acceleration(float32), jerk(float32, speed of acceleration)
		# )
		for topic, AckermannDriveMsgTuple, timestamp in bag.read_messages(topics=BAG_TOPIC):
			if rospy.is_shutdown():
				break
			seq += 1

			###		Real Car 	###
			# PUB_RATE changes over time in reality, so get the actual average hz
			# PUB_RATE = rostopic.ROSTopicHz(-1).get_hz(BAG_TOPIC)[0] # Unsuccessful! Needs to be resolved
            
			# # Figure 8, Real Car
            # percnt = 0.5
            # if seq < msg_cnt * percnt:
            #     incrm = 0.4
            # else:
            #     incrm = 0.1

            # # ECE basement
            # percnt = 0.42
            # if seq < msg_cnt * percnt:
            #     incrm = 0.03
            # else:
            #     incrm = 0.17
			# 
            # PUB_RATE = INIT_PUB_RATE + seq * incrm
			# rate = rospy.Rate(PUB_RATE)

			###		Simulation		###
			rate = rospy.Rate(PUB_RATE_SIM)

			# Setup the rest of the outgoing message
			car_trajectory.header.seq = seq		
			car_trajectory.header.stamp = rospy.Time.now()
			car_trajectory.drive.steering_angle = AckermannDriveMsgTuple.drive.steering_angle
			car_trajectory.drive.steering_angle_velocity = AckermannDriveMsgTuple.drive.steering_angle_velocity
			# If follow_backwards = True, i.e. car should drive back, only speed needs to be of opposite sign
			if follow_backwards:
				car_trajectory.drive.speed = -1 * AckermannDriveMsgTuple.drive.speed
			else:
				car_trajectory.drive.speed = AckermannDriveMsgTuple.drive.speed
			car_trajectory.drive.acceleration = AckermannDriveMsgTuple.drive.acceleration
			car_trajectory.drive.jerk = AckermannDriveMsgTuple.drive.jerk

			# Publish to PUB_TOPIC outgoing message
			pub.publish(car_trajectory)
			rate.sleep()

if __name__ == '__main__':
	bag_path = None # The file path to the bag file
	follow_backwards = False # Whether or not the path should be followed backwards
	
	rospy.init_node('bag_follower', anonymous=True)
	
	# Populate param(s) with value(s) passed by launch file
	bag_path = rospy.get_param("bag_file_path")
	follow_backwards = rospy.get_param("follow_backwards")
	
	follow_bag(bag_path, follow_backwards)
