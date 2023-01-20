#!/usr/bin/env python

import collections
import sys

import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseArray, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped

import utils

# The topic to publish control commands to
PUB_TOPIC = '/car/mux/ackermann_cmd_mux/input/navigation'


'''
Follows a given plan using constant velocity and PID control of the steering angle
'''
class LineFollower:

  '''
  Initializes the line follower
    plan: A list of length T that represents the path that the robot should follow
          Each element of the list is a 3-element numpy array of the form [x,y,theta]
    pose_topic: The topic that provides the current pose of the robot as a PoseStamped msg
    plan_lookahead: If the robot is currently closest to the i-th pose in the plan,
                    then it should navigate towards the (i+plan_lookahead)-th pose in the plan
    translation_weight: How much the error in translation should be weighted in relation
                        to the error in rotation
    rotation_weight: How much the error in rotation should be weighted in relation
                     to the error in translation
    Kp: The proportional PID parameter
    Ki: The integral PID parameter
    Kd: The derivative PID parameter
    error_buff_length: The length of the buffer that is storing past error values
    speed: The speed at which the robot should travel
  '''
  def __init__(self, plan, pose_topic, plan_lookahead, translation_weight,
               rotation_weight, Kp, Ki, Kd, error_buff_length, speed, error_plot_x, error_plot_y):
    # Store the passed parameters
    self.plan = plan
    self.plan_lookahead = plan_lookahead
    # Normalize translation and rotation weights
    self.translation_weight = translation_weight / (translation_weight+rotation_weight)
    self.rotation_weight = rotation_weight / (translation_weight+rotation_weight)
    self.Kp = Kp
    self.Ki = Ki
    self.Kd = Kd
    # The error buff stores the error_buff_length most recent errors and the
    # times at which they were received. That is, each element is of the form
    # [error, time_stamp (seconds)]. For more info about the data struct itself, visit
    # https://docs.python.org/2/library/collections.html#collections.deque
    self.error_buff = collections.deque(maxlen=error_buff_length)
    self.speed = speed
    
    self.error_plot_ind = 0
    self.error_plot_x = error_plot_x
    self.error_plot_y = error_plot_y

    # Create a command publisher to PUB_TOPIC
    self.cmd_pub = rospy.Publisher(PUB_TOPIC, AckermannDriveStamped, queue_size=10) 
    # Create a subscriber to pose_topic, with callback 'self.pose_callback'
    self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback)
  
  '''
  Computes the error based on the current pose of the car
    cur_pose: The current pose of the car, represented as a numpy array [x,y,theta]
  Returns: (False, 0.0) if the end of the plan has been reached. Otherwise, returns
           (True, E) - where E is the computed error
  '''
  def compute_error(self, cur_pose):
    
    # Find the first element of the plan that is in front of the robot, and remove
    # any elements that are behind the robot. To do this:
    # Loop over the plan (starting at the beginning) For each configuration in the plan
        # If the configuration is behind the robot, remove it from the plan
        #   Will want to perform a coordinate transformation to determine if 
        #   the configuration is in front or behind the robot
        # If the configuration is in front of the robot, break out of the loop
    while len(self.plan) > 0:
      # Compute inner product of car's heading vector (cos(theta), sin(theta)) and pointing vector (plan_x[0] - car_x, plan_y[0] - car_y)
      vect_inn_prod = (self.plan[0][0] - cur_pose[0]) * np.cos(cur_pose[2]) + (self.plan[0][1] - cur_pose[1]) * np.sin(cur_pose[2])
      # If this inner product is: negative, then car passed by the 1st element of the plan; zero, then car is starting its movement 
      if vect_inn_prod <= 0:
        self.plan.pop(0)
      else: break
      
    # Check if the plan is empty. If so, return (False, 0.0)
    if len(self.plan) == 0:
      return False, 0.0

    # At this point, we have removed configurations from the plan that are behind
    # the robot. Therefore, element 0 is the first configuration in the plan that is in 
    # front of the robot. To allow the robot to have some amount of 'look ahead',
    # we choose to have the robot head towards the configuration at index 0 + self.plan_lookahead
    # We call this index the goal_index
    goal_idx = min(0+self.plan_lookahead, len(self.plan)-1)
   
    # Compute the translation error between the robot and the configuration at goal_idx in the plan
    translation_error = -np.sin(cur_pose[2]) * (self.plan[goal_idx][0] - cur_pose[0]) + np.cos(cur_pose[2]) * (self.plan[goal_idx][1] - cur_pose[1])

    # Compute the total error
    # Translation error was computed above
    # Rotation error is the difference in yaw between the robot and goal configuration
    #   Be carefult about the sign of the rotation error
    rotation_error = np.arctan2(self.plan[goal_idx][1] - cur_pose[1], self.plan[goal_idx][0] - cur_pose[0]) - cur_pose[2] 
    error = self.translation_weight * translation_error + self.rotation_weight * rotation_error

    return True, error
    
    
  '''
  Uses a PID control policy to generate a steering angle from the passed error
    error: The current error
  Returns: The steering angle that should be executed
  '''
  def compute_steering_angle(self, error):
    now = rospy.Time.now().to_sec() # Get the current time
    
    # Compute the derivative error using the passed error, the current time,
    # the most recent error stored in self.error_buff, and the most recent time
    # stored in self.error_buff
    if len(self.error_buff) != 0:
      deriv_error = (error - self.error_buff[-1][0]) / (now - self.error_buff[-1][0])
    else: 
      deriv_error = 0.0
    
    # Add the current error to the buffer
    self.error_buff.append((error, now))
    
    # Compute the integral error by applying rectangular integration to the elements
    # of self.error_buff: https://chemicalstatistician.wordpress.com/2014/01/20/rectangular-integration-a-k-a-the-midpoint-rule/
    integ_error = 0
    for i in range(len(self.error_buff) - 1):
      integ_error += (self.error_buff[i+1][1] - self.error_buff[i][1]) * (self.error_buff[i][0] + self.error_buff[i+1][0]) / 2
    
    # Compute the steering angle as the sum of the pid errors
    return self.Kp*error + self.Ki*integ_error + self.Kd * deriv_error
    
  '''
  Callback for the current pose of the car
    msg: A PoseStamped representing the current pose of the car
    This is the exact callback that we used in our solution, but feel free to change it
  '''  
  def pose_callback(self, msg):
    cur_pose = np.array([msg.pose.position.x,
                         msg.pose.position.y,
                         utils.quaternion_to_angle(msg.pose.orientation)])
    success, error = self.compute_error(cur_pose)
    
    if not success:
      # We have reached our goal
      self.pose_sub.unregister() # Kill the subscriber
      self.speed = 0.0 # Set speed to zero so car stops

      # with open("test5.npy", "wb") as plot_file:
      #   np.save(plot_file, np.array(self.error_plot_x))
      #   np.save(plot_file, np.array(self.error_plot_y))

      with open("test1.npy", "rb") as plot_file:
        error_plot1_x = np.load(plot_file)
        error_plot1_y = np.load(plot_file)
      with open("test2.npy", "rb") as plot_file:
        error_plot2_x = np.load(plot_file)
        error_plot2_y = np.load(plot_file)
      with open("test3.npy", "rb") as plot_file:
        error_plot3_x = np.load(plot_file)
        error_plot3_y = np.load(plot_file)
      with open("test4.npy", "rb") as plot_file:
        error_plot4_x = np.load(plot_file)
        error_plot4_y = np.load(plot_file)
      with open("test5.npy", "rb") as plot_file:
        error_plot5_x = np.load(plot_file)
        error_plot5_y = np.load(plot_file)
  
      plt.title("Error graphs")
      plt.xlabel("iteration")
      plt.ylabel("error")
      plt.plot(error_plot1_x, error_plot1_y, color="red", label="plan_lookahead = 7, rotational_weight = 0.0, Ki = 0.0")
      plt.plot(error_plot2_x, error_plot2_y, color="green", label="plan_lookahead = 3, rotational_weight = 0.0, , Ki = 0.0")
      plt.plot(error_plot3_x, error_plot3_y, color="blue", label="plan_lookahead = 7, rotational_weight = 0.2, , Ki = 0.0")
      plt.plot(error_plot4_x, error_plot4_y, color="yellow", label="plan_lookahead = 7, rotational_weight = 0.4, , Ki = 0.0")
      plt.plot(error_plot5_x, error_plot5_y, color="black", label="plan_lookahead = 7, rotational_weight = 0.2, , Ki = 0.1")
      plt.legend()
      plt.show()

    delta = self.compute_steering_angle(error)

    self.error_plot_x.append(self.error_plot_ind)
    self.error_plot_ind += 1
    self.error_plot_y.append(error)
    

    # Setup the control message
    ads = AckermannDriveStamped()
    ads.header.frame_id = '/map'
    ads.header.stamp = rospy.Time.now()
    ads.drive.steering_angle = delta
    ads.drive.speed = self.speed
    
    # Send the control message
    self.cmd_pub.publish(ads)


def main():

  rospy.init_node('line_follower', anonymous=True) # Initialize the node
  
  # Load these parameters from launch file
  # We provide suggested starting values of params, but you should
  # tune them to get the best performance for your system
  # Look at constructor of LineFollower class for description of each var
  # 'Default' values are ones that probably don't need to be changed (but you could for fun)
  # 'Starting' values are ones you should consider tuning for your system

  plan_topic = rospy.get_param("plan_topic")
  pose_topic = rospy.get_param("pose_topic")
  plan_lookahead = rospy.get_param("~plan_lookahead")
  translation_weight = rospy.get_param("~translation_weight")
  rotation_weight = rospy.get_param("~rotation_weight")
  Kp = rospy.get_param("~Kp")
  Ki = rospy.get_param("~Ki")
  Kd = rospy.get_param("~Kd")
  error_buff_length = rospy.get_param("~error_buff_length")
  speed = rospy.get_param("~speed")

  # # Waits for ENTER key press
  # raw_input("Press Enter to when plan available...") 
  # # Use rospy.wait_for_message to get the plan msg
  # plan_msg = rospy.wait_for_message(plan_topic, PoseArray, 5.0)
  # rospy.loginfo("No error")
  # # Convert the plan msg, a list of Pose Messages: Pose positon(x,y,z), Quaternion orientation(x,y,z,w) 
  # # to a list of 3-element numpy arrays
  # # Each array is of the form [x,y,theta]
  # plan = []
  # for msg_cnt in range(len(plan_msg.poses)):
  #   plan.append(np.array([plan_msg.poses[msg_cnt].position.x, plan_msg.poses[msg_cnt].position.y, 
  #                         utils.quaternion_to_angle(plan_msg.poses[msg_cnt].orientation)]))
  
  # error_plot_x = []
  # error_plot_y = []
  # # Create a LineFollower object
  # line_follower = LineFollower(plan, pose_topic, plan_lookahead, translation_weight,
  #                             rotation_weight, Kp, Ki, Kd, error_buff_length, speed, error_plot_x, error_plot_y)
  
  # rospy.spin()
                    
  while not rospy.is_shutdown():
    raw_input("Press Enter to when plan available...\n")  # Waits for ENTER key press
    rospy.loginfo("Enter is pressed")

    try:
      # Use rospy.wait_for_message to get the plan msg
      plan_msg = rospy.wait_for_message(plan_topic, PoseArray, 5.0)
      rospy.loginfo("Message is received")
    except rospy.ROSException:
      rospy.loginfo("Timeout is exceeded")
      pass
    else:
      rospy.loginfo("No error")
      # Convert the plan msg, a list of Pose Messages: Pose positon(x,y,z), Quaternion orientation(x,y,z,w) 
      # to a list of 3-element numpy arrays
      # Each array is of the form [x,y,theta]
      plan = []
      for msg_cnt in range(len(plan_msg.poses)):
        plan.append(np.array([plan_msg.poses[msg_cnt].position.x, plan_msg.poses[msg_cnt].position.y, 
                              utils.quaternion_to_angle(plan_msg.poses[msg_cnt].orientation)]))

      error_plot_x = []
      error_plot_y = []
      # Create a LineFollower object
      line_follower = LineFollower(plan, pose_topic, plan_lookahead, translation_weight,
                                  rotation_weight, Kp, Ki, Kd, error_buff_length, speed, error_plot_x, error_plot_y)
    

if __name__ == '__main__':
  main()
