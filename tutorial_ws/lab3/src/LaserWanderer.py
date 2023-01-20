#!/usr/bin/env python

import rospy
import numpy as np
import math
import sys

import utils

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, Pose

SCAN_TOPIC = '/car/scan' # The topic to subscribe to for laser scans
CMD_TOPIC = '/car/mux/ackermann_cmd_mux/input/navigation' # The topic to publish controls to
POSE_TOPIC = '/car/car_pose' # The topic to subscribe to for current pose of the car
# NOTE THAT THIS IS ONLY NECESSARY FOR VIZUALIZATION
VIZ_TOPIC = '/laser_wanderer/rollouts' # The topic to publish to for vizualizing
                                       # the computed rollouts. Publish a PoseArray.
MAP_TOPIC = '/static_map' # The service topic that will provide the map

MAX_PENALTY = 10000 # The penalty to apply when a configuration in a rollout
                    # goes beyond the corresponding laser scan
                    

'''
Wanders around using minimum (steering angle) control effort while avoiding crashing
based off of laser scans. 
'''
class LaserWanderer:

  '''
  Initializes the LaserWanderer
    speed: The speed at which the car should travel
    min_delta: The minimum allowed steering angle (radians)
    max_delta: The maximum allowed steering angle (radians)
    delta_incr: The difference (in radians) between subsequent possible steering angles
    dt: The amount of time to apply a control for
    T: The number of time steps to rollout for
    compute_time: The amount of time (in seconds) we can spend computing the cost
    laser_offset: How much to shorten the laser measurements
    car_length: The length of the car
  '''
  def __init__(self, speed, min_delta, max_delta, traj_nums, dt, T, compute_time, laser_offset, car_length, car_width):
    # Store the params for later
    rospy.loginfo("2.Initialize Laser Wanderer Instance")
    self.speed = speed
    self.min_delta = min_delta    
    self.max_delta = max_delta
    self.traj_nums = traj_nums
    self.dt = dt 
    self.T = T
    self.compute_time = compute_time
    self.laser_offset = laser_offset
    self.car_length = car_length
    self.car_width = car_width

    '''
    Generate the rollouts, deltas
      rollouts: An NxTx3 numpy array that contains N rolled out trajectories, each
                containing T poses. For each trajectory, the t-th element represents
                the [x,y,theta] pose of the car at time t+1
      deltas: An N dimensional array containing the possible steering angles. The n-th
              element of this array is the steering angle that would result in the 
              n-th trajectory in rollouts
    '''
    rospy.loginfo("3.Compute Rollouts and Deltas")
    self.rollouts, self.deltas = self.generate_mpc_rollouts(speed, min_delta, max_delta, 
                                                            traj_nums, dt, T, car_length)
    rospy.loginfo("4.Create Publisher and Subscriber")
    self.cmd_pub = rospy.Publisher(CMD_TOPIC, AckermannDriveStamped, queue_size=10) # Create a publisher for sending controls
    self.laser_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self.wander_callback) # Create a subscriber to laser scans that uses the self.wander_callback
    self.viz_pub = rospy.Publisher(VIZ_TOPIC, PoseArray, queue_size=10) # Create a publisher for vizualizing trajectories. Will publish PoseArrays
    self.viz_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, self.viz_sub_callback) # Create a subscriber to the current position of the car
    # NOTE THAT THIS VIZUALIZATION WILL ONLY WORK IN SIMULATION. Why?
    
  '''
  Vizualize the rollouts. Transforms the rollouts to be in the frame of the world.
  Only display the last pose of each rollout to prevent lagginess
    msg: A PoseStamped representing the current pose of the car
  '''  
  def viz_sub_callback(self, msg):
  
    # Create the PoseArray to publish. Will contain N poses, where the n-th pose
    # represents the last pose in the n-th trajectory
    pose_array = PoseArray()
    pose_array.header.frame_id = '/map'
    pose_array.header.stamp = rospy.Time.now()

    # Transform the last pose of each trajectory to be w.r.t the world and insert into
    # the pose array
    traj_pose = Pose()
    traj_poses = []
    for trj_num, trj_point in enumerate(self.rollouts):
      vect = np.array([[trj_point[-1, 0]], [trj_point[-1, 1]]]) 
      theta_point = trj_point[-1, 2]
      theta_car = utils.quaternion_to_angle(msg.pose.orientation)
      traj_pose.position.x = utils.rotation_matrix(theta_car).dot(vect)[0] + msg.pose.position.x
      traj_pose.position.y = utils.rotation_matrix(theta_car).dot(vect)[1] + msg.pose.position.y
      traj_pose.orientation = utils.angle_to_quaternion(theta_point + theta_car)
      traj_poses.append(traj_pose)
    pose_array.poses = traj_poses

    self.viz_pub.publish(pose_array)
    
  '''
  Compute the cost of one step in the trajectory. It should penalize the magnitude
  of the steering angle. It should also heavily penalize crashing into an object
  (as determined by the laser scans)
    delta: The steering angle that corresponds to this trajectory
    rollout_pose: The pose in the trajectory 
    laser_msg: The most recent laser scan
      sensor_msgs/LaserScan Message:
      std_msgs/Header header
      float32 angle_min, start angle of the scan [rad], -pi
      float32 angle_max, end angle of the scan [rad], pi
      float32 angle_increment, angular distance between measurements [rad], 2pi/720
      float32 time_increment, time between measurements [seconds] - if your scanner
                              is moving, this will be used in interpolating position
                              of 3d points, 0.0 if not move
      float32 scan_time, time between scans [seconds]
      float32 range_min, minimum range value [m]
      float32 range_max, maximum range value [m]
      float32[720] ranges, range data [m] (Note: values < range_min or > range_max should be discarded)
      float32[] intensities, intensity data [device-specific units].  If your
                              device does not provide intensities, please leave
                              the array empty. Our laser doesn't provide this
  '''  
  def compute_cost(self, delta, rollout_pose, laser_msg):
    # NOTE THAT NO COORDINATE TRANSFORMS ARE NECESSARY INSIDE OF THIS FUNCTION
    
    # *** Without Car Physical Dimensions ***
    
    # Initialize the cost to be the magnitude of delta
    cost = np.abs(delta)
    # print("4.3.1.Trajectory Initial Cost: ", cost)

    # Compute the vector's length that goes from the robot to the rollout pose, simulation
    trj_pnt_vct_lngth = np.sqrt(np.square(rollout_pose[0]) + np.square(rollout_pose[1]))
    # print("4.3.2.Trajectory Point Distance: ", trj_pnt_vct_lngth)

    # Compute the angle (radians) of this line with respect to the robot's x axis, i.e, its orientation vector
    dtheta = np.arctan2(rollout_pose[1], rollout_pose[0])
    # print("4.3.3.dthetha: ", dtheta)rollout_pose[2]

    # Find the laser ray that corresponds to this angle.
    if not np.isfinite(dtheta):
      dtheta = np.pi / 2
      if rollout_pose[1] < 0:
        dtheta *= -1

    # Get integral part
    idx = int((dtheta - laser_msg.angle_min) / laser_msg.angle_increment)
    # print("4.3.4.Index: ", idx)
    # print("4.3.5.Laser Measurement: ", laser_msg.ranges[idx])

    laser_measurement = laser_msg.ranges[idx]

    # Add MAX_PENALTY to the cost if the distance from the robot to the rollout_pose 
    # is greater than the laser ray measurement - np.abs(self.laser_offset
    cost -= 100 * trj_pnt_vct_lngth
    if ((laser_measurement - np.abs(self.laser_offset)) <= trj_pnt_vct_lngth) and np.isfinite(laser_measurement):
      # print("4.3.6.MAX_PENALTY is applied")
      cost += MAX_PENALTY
    '''
    # *** With Car Physical Dimensions ***

    cost = np.abs(delta)

    # Compute the vector's length that goes from the robot to the rollout pose + car's front right and left edges
    Xc, Yc, theta = rollout_pose[:]
    car_frntR_edge_vct = np.array([[self.car_length], [-self.car_width/2]])
    car_frntL_edge_vct = np.array([[self.car_length], [self.car_width/2]])
    XR = Xc + utils.rotation_matrix(theta).dot(car_frntR_edge_vct)[0]
    YR = Yc + utils.rotation_matrix(theta).dot(car_frntR_edge_vct)[1]
    XL = Xc + utils.rotation_matrix(theta).dot(car_frntL_edge_vct)[0]
    YL = Yc + utils.rotation_matrix(theta).dot(car_frntL_edge_vct)[1]
    trj_pnt_vct_lngth_R = np.sqrt(np.square(XR) + np.square(YR))
    trj_pnt_vct_lngth_L = np.sqrt(np.square(XL) + np.square(YL))

    dthetaR = np.arctan2(YR, XR)
    dthetaL = np.arctan2(YL, XL)

    if not np.isfinite(dthetaR):
      dthetaR = np.pi / 2
      if YR < 0:
        dthetaR *= -1
    if not np.isfinite(dthetaL):
      dthetaL = np.pi / 2
      if YL < 0:
        dthetaL *= -1

    idxR = int((dthetaR - laser_msg.angle_min) / laser_msg.angle_increment)
    idxL = int((dthetaL - laser_msg.angle_min) / laser_msg.angle_increment)

    laser_measurementR = laser_msg.ranges[idxR]
    laser_measurementL = laser_msg.ranges[idxL]
    
    cost -= 100 * (trj_pnt_vct_lngth_L + trj_pnt_vct_lngth_R) / 2

    if ((laser_measurementR - np.abs(self.laser_offset)) <= trj_pnt_vct_lngth_R) and np.isfinite(laser_measurementR):
      # print("4.3.6.MAX_PENALTY is applied")
      cost += MAX_PENALTY
    if ((laser_measurementL - np.abs(self.laser_offset)) <= trj_pnt_vct_lngth_L) and np.isfinite(laser_measurementL):
      # print("4.3.6.MAX_PENALTY is applied")
      cost += MAX_PENALTY
    '''

    # Return the resulting cost
    return cost

    # Things to think about:
    #   What if the angle of the pose is less (or greater) than the angle of the
    #   minimum (or maximum) laser scan angle
    #   What if the corresponding laser measurement is NAN or 0?
    
    
  '''
  Controls the steering angle in response to the received laser scan. Uses approximately
  self.compute_time amount of time to compute the control
    msg: A LaserScan
  '''
  def wander_callback(self, msg):
    # if self.ray_cnt == 720:
    #   self.ray_cnt = 0
    # self.ray_cnt += 1
    # print()
    # print()
    # rospy.loginfo("4.1.Wander Callback Function is called")
    start = rospy.Time.now().to_sec() # Get the time at which this function started
    
    # A N dimensional matrix that should be populated with the costs of each
    # trajectory up to time t <= T
    delta_costs = np.zeros(self.deltas.shape[0], dtype=np.float) 
    traj_depth = 0
    
    # Evaluate cost of each trajectory. Each iteration of the loop should calculate
    # the cost of each trajectory at time t = traj_depth and add those costs to delta_costs
    # as appropriate
    while (rospy.Time.now().to_sec() - start <= self.compute_time) and (traj_depth < self.T):
      # print()
      # print("4.2.Trajectory depth: ", traj_depth)
      for traj_num, delta in enumerate(self.deltas):
          # print("4.3.Trajectory Cost is being Calculated")
          delta_costs[traj_num] += self.compute_cost(delta, self.rollouts[traj_num][traj_depth], msg)
      traj_depth += 1
    print()
    print()
    print(traj_depth)
    print(delta_costs)


    # Find index of delta that has the smallest cost and execute it by publishing
    min_cost_delta_idx = np.argmin(delta_costs, axis=0)
    print("4.4.Trajectory with Min Cost is with Index: ", min_cost_delta_idx)

    ads = AckermannDriveStamped()
    ads.header.frame_id = '/laser_link'
    ads.header.stamp = rospy.Time.now()
    ads.drive.steering_angle = self.deltas[min_cost_delta_idx]
    ads.drive.speed = self.speed
    # Publish command to drive car along selected trajectory
    self.cmd_pub.publish(ads)
    # print("4.5.Command is Published! Car is moving")

  '''
  Apply the kinematic model to the passed pose and control
    pose: The current state of the robot [x, y, theta]
    control: The controls to be applied [v, delta, dt]
    car_length: The length of the car
  Returns the resulting pose of the robot
  '''
  def kinematic_model_step(self, current_pose, control, car_length):
    # Apply the kinematic model
    # Make sure your resulting theta is between 0 and 2*pi
    # Consider the case where delta == 0.0
    next_pose = current_pose.copy()
    v = control[0]
    delta = control[1]
    dt = control[2]
    theta = current_pose[2]
    
    # Compute pose(x,y,theta) for next point in trajectory, given the current pose
    next_pose[0] += v * np.cos(theta) * dt
    next_pose[1] += v * np.sin(theta) * dt
    next_pose[2] += v * np.tan(delta) * dt / car_length
    # print(next_pose)

    return next_pose
    
  '''
  Repeatedly apply the kinematic model to produce a trajectory for the car
    init_pose: The initial pose of the robot [x,y,theta]
    controls: A Tx3 numpy matrix where each row is of the form [v,delta,dt]
    car_length: The length of the car
  Returns a Tx3 matrix where the t-th row corresponds to the robot's pose at time t+1
  '''
  def generate_rollout(self, init_pose, controls, car_length):
    rospy.loginfo("3.2.1.Start Generating Rollout for Given Trajectory")
    rollout_mat = np.zeros((controls.shape[0], 3), dtype=np.float)
    # Compute pose(x,y,theta) for each point in trajectory
    for pose_num, traj_detail in enumerate(controls):
      current_pose = init_pose.copy()
      if pose_num > 0:
        current_pose += rollout_mat[pose_num-1]
      # rospy.loginfo("Compute Kinematic Model Step")
      rollout_mat[pose_num] = self.kinematic_model_step(current_pose, traj_detail, car_length)
    rospy.loginfo("3.2.2.Rollout for Given Trajectory Generation is Completed")
    return rollout_mat
   
  '''
  Helper function to generate a number of kinematic car rollouts
      speed: The speed at which the car should travel
      min_delta: The minimum allowed steering angle (radians)
      max_delta: The maximum allowed steering angle (radians)
      delta_incr: The difference (in radians) between subsequent possible steering angles
      dt: The amount of time to apply a control for
      T: The number of time steps to rollout for
      car_length: The length of the car
  Returns a NxTx3 numpy array that contains N rolled out trajectories, each
  containing T poses. For each trajectory, the t-th element represents the [x,y,theta]
  pose of the car at time t+1
  '''
  def generate_mpc_rollouts(self, speed, min_delta, max_delta, traj_nums, dt, T, car_length):
    rospy.loginfo("3.1.Starting to Generate Rollouts")
    delta_step = (max_delta - min_delta - 0.001) / traj_nums
    deltas = np.arange(min_delta, max_delta, delta_step)
    N = deltas.shape[0]
    init_pose = np.array([0.0,0.0,0.0], dtype=np.float)

    rollouts = np.zeros((N,T,3), dtype=np.float)
    for i in xrange(N):
      controls = np.zeros((T,3), dtype=np.float)
      controls[:,0] = speed
      controls[:,1] = deltas[i]
      controls[:,2] = dt
      rospy.loginfo("3.2.Generate One Rollout for Each Trajectory")
      rollouts[i,:,:] = self.generate_rollout(init_pose, controls, car_length)
      
    return rollouts, deltas

    

def main():

  rospy.init_node('laser_wanderer', anonymous=True)
  # Inititalize global variable "car_current_pose" to catch "/car/car_pose" message 
  # as [x, y, theta] and use it "compute_cost" function

  # Load these parameters from launch file
  speed = rospy.get_param("speed")
  min_delta = rospy.get_param("min_delta")
  max_delta = rospy.get_param("max_delta")
  traj_nums = rospy.get_param("~traj_nums")
  dt = rospy.get_param("dt")
  T = rospy.get_param("~T")
  compute_time = rospy.get_param("compute_time")
  laser_offset = rospy.get_param("~laser_offset")
  
  # DO NOT ADD THIS TO YOUR LAUNCH FILE, car_length and car_width are already provided by teleop.launch
  car_length = rospy.get_param("/car/vesc/chassis_length", 0.33)
  car_width = rospy.get_param("/car/vesc/wheelbase", 0.25)
  
  # Create the LaserWanderer
  rospy.loginfo("1.1.Create Laser Wanderer Instance")                                         
  laser_wanderer = LaserWanderer(speed, min_delta, max_delta, traj_nums, dt, T, compute_time, laser_offset, car_length, car_width)
  rospy.loginfo("1.2.Laser Wanderer Instance Created!")
  
  # Keep the node alive
  rospy.spin()
  

if __name__ == '__main__':
  rospy.loginfo("Start!")
  main()
