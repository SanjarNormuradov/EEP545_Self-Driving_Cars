#!/usr/bin/env python

import rospy, math, sys, Utils
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, Pose

CMD_TOPIC = '/car/mux/ackermann_cmd_mux/input/navigation' # The topic to publish controls to
MAP_TOPIC = '/static_map' # The service topic that will provide the map

# NOTE THAT THIS IS ONLY NECESSARY FOR VIZUALIZATION
VIZ_TOPIC = '/mpcontroller/rollouts' # The topic to publish to for vizualizing
                                       # the computed rollouts. Publish a PoseArray.

REWARD = 10 # The reward to apply when a point in a rollout is close to a point in the plan
                    

'''
Wanders around using minimum (steering angle) control effort while avoiding crashing
based off of laser scans. 
'''
class MPController:

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
  def __init__(self, plan, pose_topic, min_speed, max_speed, min_delta, max_delta, 
                     traj_nums, dt, T, compute_time, car_length, car_width, visual_sim=True):
    # Store the params for later
    print "[MPC Controller] Initialization..."
    self.plan = plan
    print "[MPC Controller] Plan length: %d" % len(self.plan)
    self.T = T
    self.compute_time = compute_time
    self.car_length = car_length
    self.car_width = car_width
    self.visual_sim = visual_sim
    self.success = False
    self.first_indx = 0

    '''
    Generate the rollouts, deltas
      rollouts: An NxTx3 numpy array that contains N rolled out trajectories, each
                containing T poses. For each trajectory, the t-th element represents
                the [x,y,theta] pose of the car at time t+1
      deltas: An N dimensional array containing the possible steering angles. The n-th
              element of this array is the steering angle that would result in the 
              n-th trajectory in rollouts
    '''
    print "[MPC Controller]  Rollouts and deltas computation..."
    self.rollouts, self.deltas, self.speeds = self.generate_mpc_rollouts(min_speed, max_speed, min_delta, max_delta, 
                                                            traj_nums, dt, T, car_length)
    print "[MPC Controller] Rollouts and deltas computation complete"
    print "[MPC Controller] Deltas:\n%s" % str(self.deltas)
    print "[MPC Controller] Speeds:\n%s" % str(self.speeds)
    print "[MPC Controller] Rollouts:\n%s" % str(self.rollouts)
    raw_input("[MPC Controller] Press Enter\n")
    self.cmd_pub = rospy.Publisher(CMD_TOPIC, AckermannDriveStamped, queue_size=10) # Create a publisher for sending controls
    if self.visual_sim:
      # NOTE THAT THIS VISUALIZATION WORKS ONLY IN SIMULATION.
      self.viz_pub = rospy.Publisher(VIZ_TOPIC, PoseArray, queue_size=10) # Create a publisher for vizualizing trajectories. Will publish PoseArrays
    self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback) # Create a subscriber to the current position of the car
    print "[MPC Controller] Initialization complete"
    

  '''
  Controls the steering angle in response to the received laser scan. Uses approximately self.compute_time amount of time to compute the control
  If self.visual_sim == True: 
    Vizualize the rollouts. Transforms the rollouts to be in the frame of the world.
    Only display the last pose of each rollout to prevent lagginess
  msg: The most recent car pose
    geometry_msgs/PoseStamped Message:
    std_msgs/Header header
    geometry_msgs/Pose pose:
      geometry_msgs/Point position:
                          float64 x
                          float64 y
                          float64 z
      geometry_msgs/Quaternion orientation  
  '''
  def pose_callback(self, msg):
    # Get the time at which this function started
    start = rospy.Time.now().to_sec()
    cur_pose = np.array([msg.pose.position.x,
                         msg.pose.position.y,
                         Utils.quaternion_to_angle(msg.pose.orientation)])

    # Find index of the first point in the plan that is in front of the robot,
    # DO NOT REMOVE points behind the robot as in PID controller, because we compute cost for each trajectory
    # Loop over the plan (starting from the index that was found in the previous step) 
    # For each point in the plan:
    #   If the point is behind the robot, increase the index
    #     Perform a coordinate transformation to determine if the point is in front or behind the robot
    #   If the point is in front of the robot, break out of the loop
    while self.first_indx < len(self.plan):
      # print "[MPC Controller] First front point index: %d" % self.first_indx
      # Compute inner product of car's heading vector (cos(theta), sin(theta)) and pointing vector (plan_x[0] - car_x, plan_y[0] - car_y)
      pntVect = self.plan[self.first_indx][0:2] - cur_pose[0:2] 
      pntVectNorm = np.sqrt(np.power(pntVect, 2).sum())
      pntVect /= pntVectNorm
      vect_inn_prod = pntVect[0] * np.cos(cur_pose[2]) + pntVect[1] * np.sin(cur_pose[2])
      # If this inner product is: negative, then car passed by the 1st element of the plan; zero, then car is starting its movement 
      if vect_inn_prod <= 0:
        # print "[MPC Controller] Plan first front point is behind"
        self.first_indx += 1
      else: break
    # Check if the index is over the plan's last index. If so, stop movement
    if self.first_indx == len(self.plan):
      # print "[MPC Controller] Target is achieved!"
      self.success = True
      self.pose_sub.unregister() # Kill the subscriber

    print "[MPC Controller] First index: %d" % self.first_indx
    if not self.success:
      # N-dimensional matrix that should be populated with the costs of each trajectory up to time t <= T
      delta_costs = np.zeros(self.deltas.shape[0], dtype=np.float) 
      traj_depth = 0
      # Evaluate cost of each trajectory. Each iteration of the loop should calculate
      # the cost of each trajectory at time t = traj_depth and add those costs to delta_costs as appropriate
      while (rospy.Time.now().to_sec() - start <= self.compute_time) and (traj_depth < self.T):
        for traj_num, delta in enumerate(self.deltas):
          delta_costs[traj_num] += self.compute_cost(delta, traj_depth, self.rollouts[traj_num, traj_depth], cur_pose)
        traj_depth += 1
      print "[MPC Controller] Trajectory depth: %d" % traj_depth
      # Find index of delta that has the smallest cost and execute it by publishing
      min_cost_delta_idx = np.argmin(delta_costs, axis=0)
      print "[MPC Controller] Delta_costs: %s" % str(delta_costs)
      print "[MPC Controller] Min cost trajectory index: %d" % min_cost_delta_idx

    # ads.header.frame_id = '/laser_link'
    # Setup the control message
    ads = AckermannDriveStamped()
    ads.header = Utils.make_header("/map")
    ads.drive.steering_angle = 0.0 if self.success else self.deltas[min_cost_delta_idx]
    ads.drive.speed = 0.0 if self.success else self.speeds[min_cost_delta_idx]
    # Send the control message
    self.cmd_pub.publish(ads)

    if self.visual_sim:
      # Create the PoseArray to publish. Will contain N poses, where the n-th pose represents the last pose in the n-th trajectory
      pose_array = PoseArray()
      pose_array.header = Utils.make_header("/map")
      # Transform the last pose of each trajectory to be w.r.t the world and insert into the pose array
      traj_pose = Pose()
      for trj_num, trj in enumerate(self.rollouts):
        vect = np.array([[trj[-1, 0]], [trj[-1, 1]]]) 
        theta_point = trj[-1, 2]
        theta_car = cur_pose[2]
        traj_pose.position.x = Utils.rotation_matrix(theta_car).dot(vect)[0] + cur_pose[0]
        traj_pose.position.y = Utils.rotation_matrix(theta_car).dot(vect)[1] + cur_pose[1]
        traj_pose.orientation = Utils.angle_to_quaternion(theta_point + theta_car)
        pose_array.poses.append(traj_pose)
      self.viz_pub.publish(pose_array)


  '''
  Compute the cost of one step in the trajectory. It should penalize the magnitude
  of the steering angle. It should also heavily penalize crashing into an object
  (as determined by the laser scans)
    delta: The steering angle that corresponds to this trajectory
    rollout_pose: The pose in the trajectory 

  '''  
  def compute_cost(self, delta, traj_depth, rollout_pose, cur_pose):
    # NOTE THAT NO COORDINATE TRANSFORMS ARE NECESSARY INSIDE OF THIS FUNCTION

    # Computation without car's physical dimensions, because A* planner accounted the dimensions during the path generation
    # Initialize the cost to be the magnitude of delta
    cost = np.abs(delta)

    # Compute the cost of following each trajectory, 
    # i.e. inner product of two vectors directed from current robot's pose to:
    #     1. trajectory point with traj_depth
    #     2. plan point with index (self.first_indx + traj_depth)
    planVect = self.plan[self.first_indx + traj_depth][:2] - cur_pose[:2]
    planVectNorm = np.sqrt(np.power(planVect, 2).sum())
    planVect /= planVectNorm

    trajVect = rollout_pose[:2] - cur_pose[:2]
    trajVectNorm = np.sqrt(np.power(trajVect, 2).sum())
    trajVect /= trajVectNorm
      
    # print "[MPC Controller] Cost computation..."
    cost = -(trajVect[0] * planVect[0] + trajVect[1] * planVect[1]) * REWARD
    # Return the resulting cost
    return cost
    

  '''
  Apply the kinematic model to the passed pose and control
    pose: The current state of the robot [x,y,theta]
    control: The controls to be applied [v,delta,dt]
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
    rollout_mat = np.zeros((controls.shape[0], 3), dtype=np.float)
    # Compute pose(x,y,theta) for each point in trajectory
    for pose_num, traj_detail in enumerate(controls):
      current_pose = init_pose.copy()
      if pose_num > 0:
        current_pose += rollout_mat[pose_num-1]
      rollout_mat[pose_num] = self.kinematic_model_step(current_pose, traj_detail, car_length)
    return rollout_mat
   
  '''
  Helper function to generate a number of kinematic car rollouts
      min_speed | max_speed: The speed at which the car should travel. More straight line trajectory is, more speed
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
  def generate_mpc_rollouts(self, min_speed, max_speed, min_delta, max_delta, 
                                  traj_nums, dt, T, car_length):
    delta_step = (max_delta - min_delta - 0.001) / (traj_nums - 1)
    deltas = np.arange(min_delta, max_delta, delta_step)
    speeds = np.zeros_like(deltas, dtype=np.float)
    N = deltas.shape[0]
    init_pose = np.array([0.0, 0.0, 0.0], dtype=np.float)

    rollouts = np.zeros((N,T,3), dtype=np.float)
    for i in xrange(N):
      speeds[i] = max_speed - ((max_speed - min_speed) / ((N-1)/2)) * abs((N-1)/2 - i)
      controls = np.zeros((T,3), dtype=np.float)
      controls[:,0] = speeds[i]
      controls[:,1] = deltas[i]
      controls[:,2] = min_speed / speeds[i] * dt
      # print "[MPC Controller] Rollout %d generation..." % i
      rollouts[i,:,:] = self.generate_rollout(init_pose, controls, car_length)
      # print "[MPC Controller] Rollout %d generation complete" % i
    return rollouts, deltas, speeds
