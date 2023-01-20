#!/usr/bin/env python

import numpy as np
import rospy, time
import range_libc
import math
from threading import Lock
from nav_msgs.srv import GetMap
import rosbag
import matplotlib.pyplot as plt
import utils as Utils
from sensor_msgs.msg import LaserScan


MAP_TOPIC = 'static_map'
THETA_DISCRETIZATION = 112  # Discretization of scanning angle
INV_SQUASH_FACTOR = 0.2  # Factor for helping the weight distribution to be less peaked

# Sensor Model Intrinsic Parameters. Z_SHORT + Z_MAX + Z_RAND + Z_HIT = 1
Z_SHORT = 0.05  # Weight for short reading, 0.2 (Yao)
Z_MAX = 0.04  # Weight for max reading, 0.15 (Yao)
Z_RAND = 0.01  # Weight for random reading, 0.1 (Yao)
Z_HIT = 0.9  # Weight for hit reading, 0.55 (Yao)
SIGMA_HIT = 10.5  # Noise value for hit reading, 1.3 (Yao)
LAMBDA_SHORT = 0.001  # Exponential Distribution, 0.001 (Yao)


''' 
  Weights particles according to their agreement with the observed data
'''
class SensorModel:
  '''
  Initializes the sensor model
    scan_topic: The topic containing laser scans
    laser_ray_step: Step for downsampling laser scans
    exclude_max_range_rays: Whether to exclude rays that are beyond the max range
    max_range_meters: The max range of the laser
    map_msg: A nav_msgs/MapMetaData msg containing the map to use
    particles: The particles to be weighted
    weights: The weights of the particles
    state_lock: Used to control access to particles and weights
  '''
  def __init__(self, scan_topic, laser_ray_step, exclude_max_range_rays, 
               max_range_meters, map_msg, particles, weights, state_lock=None):
    print "[Sensor Model] Initialization..."
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock

    self.particles = particles
    self.weights = weights
    self.last_laser = None

    self.LASER_RAY_STEP = laser_ray_step  # Step for downsampling laser scans
    self.EXCLUDE_MAX_RANGE_RAYS = exclude_max_range_rays  # Whether to exclude rays that are beyond the max range
    self.MAX_RANGE_METERS = max_range_meters  # The max range of the laser

    oMap = range_libc.PyOMap(map_msg)  # A version of the map that range_libc can understand
    max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution)  # The max range in pixels of the laser
    self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, THETA_DISCRETIZATION)  # The range method that will be used for ray casting
    # self.range_method = range_libc.PyRayMarchingGPU(oMap, max_range_px) # The range method that will be used for ray casting
    self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px))  # Load the sensor model expressed as a table
    self.queries = None  # Do not modify this variable
    self.ranges = None  # Do not modify this variable
    self.downsampled_ranges = None  # The ranges of the downsampled rays
    self.downsampled_angles = None  # The angles of the downsampled rays
    self.do_resample = False  # Set so that outside code can know that it's time to resample

    # Subscribe to laser scans
    self.laser_sub = rospy.Subscriber(scan_topic, LaserScan, self.lidar_callback, queue_size=1)
    print "[Sensor Model] Initialization complete"


  '''
    Downsamples laser measurements and applies sensor model
      msg: A sensor_msgs/LaserScan
  '''
  def lidar_callback(self, msg):
    self.state_lock.acquire()

    # Compute the observation obs
    #   obs is a a two element tuple
    #   obs[0] is the downsampled ranges
    #   obs[1] is the downsampled angles
    #   Note it should be the case that obs[0].shape[0] == obs[1].shape[0]
    #   Each element of obs must be a numpy array of type np.float32
    #   Use self.LASER_RAY_STEP as the downsampling step
    #   Keep efficiency in mind, including by caching certain things that won't change across future iterations of this callback
    #                                                                           and vectorizing computations as much as possible
    #   Set all range measurements that are NAN or 0.0 to self.MAX_RANGE_METERS
    #   You may choose to use self.downsampled_ranges and self.downsampled_angles here

    if self.downsampled_angles == None:
      self.downsample_size = len(msg.ranges) // self.LASER_RAY_STEP
      self.downsampled_angles = []
      for angle_idx in xrange(self.downsample_size):
        self.downsampled_angles.append(msg.angle_min + msg.angle_increment * angle_idx * self.LASER_RAY_STEP)
    
    self.downsampled_ranges = []
    for idx in xrange(self.downsample_size):
      obs_range = msg.ranges[idx * self.LASER_RAY_STEP]
      if np.isnan(obs_range) or obs_range == 0.0:
        self.downsampled_ranges.append(self.MAX_RANGE_METERS)
      else:
        self.downsampled_ranges.append(obs_range)

    obs = np.zeros((2, self.downsample_size), dtype=np.float32)
    obs[0,:] = np.array(self.downsampled_ranges, dtype=np.float32)
    obs[1,:] = np.array(self.downsampled_angles, dtype=np.float32)
    # print "[Sensor Model] Weights: %s" % str(self.weights)
    self.apply_sensor_model(self.particles, obs, self.weights)
    # print "[Sensor Model] Weights: %s" % str(self.weights)
    self.weights /= np.sum(self.weights)

    self.last_laser = msg
    self.do_resample = True
    self.state_lock.release()


  '''
    Compute table enumerating the probability of observing a measurement 
    given the expected measurement
    Element (r,d) of the table is the probability of observing (true) measurement r (in pixels)
    when the expected (simulated) measurement is d (in pixels)
    max_range_px: The maximum range in pixels
    Returns the table (which is a numpy array with dimensions [max_range_px+1, max_range_px+1]) 
  '''
  def precompute_sensor_model(self, max_range_px):
    print "[Sensor Model] Sensor model generation as table..."
    table_width = int(max_range_px) + 1
    sensor_model_table = np.zeros((table_width, table_width))

    # Populate sensor_model_table according to the laser beam model specified
    # in CH 6.3 of Probabilistic Robotics
    # Note: no need to use any functions from utils.py to compute between world
    #       and map coordinates here
    variance = np.power(SIGMA_HIT, 2)
    norm = 1
    for true_range in range(table_width):
      # Point-mass Distribution to account 'Max-range Measurments' such as specular (mirror) reflections (sonars) and black, light-absorbing, objects or measuring in bright-sunlight (lasers)
      p_max = float(true_range == int(max_range_px))
      # Uniform Distribution to account 'Unexplainable Measurements' such as cross-talk between different sensors or phantom readings bouncing off walls (sonars)
      p_rand = float(true_range < int(max_range_px)) * 1 / int(max_range_px)
      for sim_range in range(table_width):
        if true_range == 0:
          p_short = 0.0
          norm = math.erf(float(max_range_px) / (math.sqrt(2) * SIGMA_HIT)) / 2
        else:
          norm = 1 / -np.expm1(-LAMBDA_SHORT * true_range) 
          # Exponential Distribution to account 'Unexpected Objects' such as people not contained in the static map
          p_short = float(sim_range <= true_range) * norm * LAMBDA_SHORT * np.exp(-LAMBDA_SHORT * sim_range)
        # Gaussian Distribution to account 'Measurement Noise'
        p_hit = float(true_range <= int(max_range_px)) * norm * 1 / (np.sqrt(2 * np.pi * variance)) * np.exp(-1 / 2 * np.power(true_range - sim_range, 2) / variance)

        sensor_model_table[sim_range, true_range] = Z_HIT * p_hit + Z_SHORT * p_short + Z_MAX * p_max + Z_RAND * p_rand
    # Normalization over all probabilities suchas as below is not correct
    sensor_model_table /= sensor_model_table.sum(axis=0)
    print "[Sensor Model] Generation complete"
    return sensor_model_table


  '''
    Updates the particle weights in-place based on the observed laser scan
    proposal_dist: The particles
    obs: The most recent observation
    weights: The weights of each particle
  '''
  def apply_sensor_model(self, proposal_dist, obs, weights):

    obs_ranges = obs[0]
    obs_angles = obs[1]
    num_rays = self.downsample_size

    # Only allocate buffers once to avoid slowness
    if not isinstance(self.queries, np.ndarray):
      self.queries = np.zeros((proposal_dist.shape[0], 3), dtype=np.float32)
      self.ranges = np.zeros(num_rays * proposal_dist.shape[0], dtype=np.float32)

    self.queries[:,:] = proposal_dist[:,:]

    # Raycasting to get expected measurements
    self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

    # Evaluate the sensor model
    self.range_method.eval_sensor_model(obs_ranges, self.ranges, weights, num_rays, proposal_dist.shape[0])

    # Squash weights to prevent too much peakiness
    np.power(weights, INV_SQUASH_FACTOR, weights)


'''
  Code for testing SensorModel
'''
if __name__ == '__main__':
  rospy.init_node("sensor_model", anonymous=True)  # Initialize the node

  bag_path = rospy.get_param("~bag_path", "/home/robot/tutorial_ws/src/lab4/bags/laser_scans/laser_scan1.bag")
  scan_topic = rospy.get_param("~scan_topic", "/scan")  # The topic containing laser scans
  laser_ray_step = rospy.get_param("~laser_ray_step", 20)  # Step for downsampling laser scans
  exclude_max_range_rays = rospy.get_param("~exclude_max_range_rays", "True")  # Whether to exclude rays that are beyond the max range
  max_range_meters = rospy.get_param("~max_range_meters", 10.0)  # The max range of the laser

  print "[Sensor Model] Bag path: " + bag_path
  '''
  Use the 'static_map' service (launched by MapServer.launch) 
  to get the map as nav_msgs/OccupancyGrid:
                                          std_msgs/Header header:
                                                                uint32 seq
                                                                time stamp
                                                                string frame_id
                                          nav_msgs/MapMetaData info:
                                                                    time map_load_time
                                                                    float32 resolution
                                                                    uint32 width
                                                                    uint32 height
                                                                    geometry_msgs/Pose origin
                                          int8[] data, length = 10240000 (3200 x 3200)
  '''
  print "[Sensor Model] Getting map from service: " + MAP_TOPIC
  rospy.wait_for_service(MAP_TOPIC)
  map_msg = rospy.ServiceProxy(MAP_TOPIC, GetMap)().map  # The map, will get passed to init of sensor model
  map_info = map_msg.info  # Save info about map for later use
  print "[Sensor Model] ...got map"

  print "[Sensor Model] Creating permissible regions..."
  # Create numpy array representing map for later use
  array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
  print "[Sensor Model] Map: Height: %d, Weight: %d" % (map_msg.info.height, map_msg.info.width)

  permissible_region = np.zeros_like(array_255, dtype=bool)
  permissible_region[array_255 == 0] = 1  # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
  # With values 0: not permissible, 1: permissible
  permissible_y, permissible_x = np.where(permissible_region == 1)
  # print "[Sensor Model] Permissable regions shape: %s" % (permissible_x.shape)
  # print "[Sensor Model] Permissable region X:\n%s" % (permissible_x)
  # print "[Sensor Model] Permissable region Y:\n%s" % (permissible_y)

  # Downsample permissible_x and permissible_y
  print "[Sensor Model] Permissable region size before downsampling: %d" % (permissible_x.shape[0])
  dsp_x = []; dsp_y = []
  ds = 4
  for i in range(permissible_x.shape[0]):
    if np.random.randint(ds) == 0:
      dsp_x.append(permissible_x[i])
      dsp_y.append(permissible_y[i])
  dsp_x = np.array(dsp_x, dtype=np.int32)
  dsp_y = np.array(dsp_y, dtype=np.int32)
  print "[Sensor Model] Permissable region size after downsampling: %d" % (dsp_x.shape[0])

  print "[Sensor Model] Creating particles and weights..."
  angle_step = 25
  particles = np.zeros((angle_step * dsp_x.shape[0], 3))
  for i in xrange(angle_step):
    particles[i * (particles.shape[0] / angle_step):(i + 1) * (particles.shape[0] / angle_step), 0] = dsp_x[:]
    particles[i * (particles.shape[0] / angle_step):(i + 1) * (particles.shape[0] / angle_step), 1] = dsp_y[:]
    particles[i * (particles.shape[0] / angle_step):(i + 1) * (particles.shape[0] / angle_step), 2] = i * (2 * np.pi / angle_step)

  Utils.map_to_world(particles, map_info)
  weights = np.ones(particles.shape[0]) / float(particles.shape[0])
  print "[Sensor Model] Particles and weights created"

  SM = SensorModel(scan_topic, laser_ray_step, exclude_max_range_rays,
                     max_range_meters, map_msg, particles, weights)

  # Give time to get setup
  rospy.sleep(1.0)

  # Load laser scan from bag
  bag = rosbag.Bag(bag_path)
  for topic, msg, timestamp in bag.read_messages(topics=['/scan']):
    laser_msg = msg
    break

  w_min = np.amin(weights)
  w_max = np.amax(weights)

  pub_laser = rospy.Publisher(scan_topic, LaserScan, queue_size=1)  # Publishes the most recent laser scan
  print "[Sensor Model] Starting analysis, this could take awhile..."
  while not isinstance(SM.queries, np.ndarray):
    pub_laser.publish(laser_msg)
    rospy.sleep(1.0)

  rospy.sleep(1.0)  # Make sure there's enough time for laserscan to get lock

  print "[Sensor Model] Going to wait for sensor model to finish..."
  SM.state_lock.acquire()
  print "[Sensor Model] Done, preparing to plot..."
  weights = weights.reshape((angle_step, -1))
  weights = np.amax(weights, axis=0)
  print "[Sensor Model] Weights Shape: %d" % (weights.shape)
  w_min = np.amin(weights)
  w_max = np.amax(weights)
  print "[Sensor Model] w_min = %f" % w_min
  print "[Sensor Model] w_max = %f" % w_max
  weights = 0.9 * (weights - w_min) / (w_max - w_min) + 0.1

  img = np.zeros((map_msg.info.height, map_msg.info.width))
  for i in xrange(len(dsp_x)):
    img[dsp_y[i], dsp_x[i]] = weights[i]
  plt.imshow(img)
  plt.show()