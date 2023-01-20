#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped # Header(), Pose(Current Position(x,y,z), Orientation(Quaternion msg))
import Utils

SUB_TOPIC = '/car/car_pose' # The topic that provides the simulated car pose
PUB_TOPIC = '/clone_follower_pose/pose' # The topic that you should publish to for clone pose
MAP_TOPIC = '/static_map' # The service topic that will provide the map

# Follows the simulated robot around
class CloneFollower:

  '''
  Initializes a CloneFollower object
  In:
    follow_offset: The required x offset between the robot and its clone follower
    force_in_bounds: Whether the clone should toggle between following in front
                     and behind when it goes out of bounds of the map
  ''' 
  def __init__(self, follow_offset, force_in_bounds):
    self.follow_offset = follow_offset # Store the input params in self
    self.force_in_bounds = force_in_bounds # Store the input params in self
    self.map_img, self.map_info = Utils.get_map(MAP_TOPIC) # Get and store the map for 
							   # bounds checking
    
    # Setup publisher that publishes to PUB_TOPIC
    self.pub = rospy.Publisher(PUB_TOPIC, PoseStamped, queue_size=10)
    
    # Setup subscriber that subscribes to SUB_TOPIC and uses the self.update_pose callback
    self.sub = rospy.Subscriber(SUB_TOPIC, PoseStamped, self.update_pose)
    
  '''
  Given the translation and rotation between the robot and map, computes the pose
  of the clone
  (This function is optional)
  In:
    trans: The translation between the robot and map
    rot: The rotation between the robot and map
  Out:
    The pose of the clone
  '''
  def compute_follow_pose(self, trans, rot):
    yaw = Utils.quaternion_to_angle(rot)
    # Create a vector [x, y] with x = follow_offset (1.5m) and y = 0
    V_follow_offset = np.array([[self.follow_offset], [0.0]])
    print(V_follow_offset)
    # Compute matrix multiplication of rotation matrix and previous vector   
    offset_matrix = np.dot(Utils.rotation_matrix(yaw), V_follow_offset)
    print(offset_matrix)
    x = trans.x + offset_matrix[0]
    y = trans.y + offset_matrix[1]
    return [x, y, yaw]

  '''
  Callback that runs each time a sim pose is received. Should publish an updated
  pose of the clone.
  In:
    msg: The pose of the simulated car. Should be a geometry_msgs/PoseStamped
  '''  
  def update_pose(self, msg):  
    # Compute the pose of the clone
    updated_clone_pose = self.compute_follow_pose(msg.pose.position, msg.pose.orientation)
    print(updated_clone_pose)

    # Check bounds if required
    if self.force_in_bounds:
      clone_world_location = Utils.world_to_map(updated_clone_pose, self.map_info)
      if not self.map_img[clone_world_location[0]][clone_world_location[1]]:
        # We hit the boundary! Change the direction, i.e. the sign of follow_offset
        self.follow_offset *= -1
        # Compute new clone position
        updated_clone_pose = self.compute_follow_pose(msg.pose.position, msg.pose.orientation)
    
    # Setup the out going PoseStamped message
    pub_clone_pose = PoseStamped()
    pub_clone_pose.header.stamp = rospy.Time.now()
    pub_clone_pose.header.frame_id = "/map"

    pub_clone_pose.pose.position.x = updated_clone_pose[0]
    pub_clone_pose.pose.position.y = updated_clone_pose[1]
    pub_clone_pose.pose.orientation = Utils.angle_to_quaternion(updated_clone_pose[2])

    # Publish the clone's pose
    self.pub.publish(pub_clone_pose)
    
if __name__ == '__main__':
  follow_offset = 1.5 # The offset between the robot and clone
  force_in_bounds = False # Whether or not map bounds should be enforced
  
  rospy.init_node('clone_follower', anonymous=True) # Initialize the node
  
  # Populate params with values passed by launch file
  follow_offset = rospy.get_param("follow_offset")
  force_in_bounds = rospy.get_param("force_in_bounds")
  
  cf = CloneFollower(follow_offset, force_in_bounds) # Create a clone follower class instance
  rospy.spin() # Spin, i.e don't close the program until it's shutdown
  
