#!/usr/bin/env python

import rospy
import numpy as np
from numpy.random import normal as Gauss
import utils as Utils
from std_msgs.msg import Float64
from threading import Lock
from nav_msgs.msg import Odometry
from vesc_msgs.msg import VescStateStamped
import matplotlib.pyplot as plt

# Set these values and use them in motion_callback
KM_V_NOISE = 0.01 # Kinematic car velocity noise, which is normally distributed with standard deviation
KM_DELTA_NOISE = 0.05 # Kinematic car delta noise, which is normally distributed with standard deviation
KM_X_FIX_NOISE = 0.01 # Kinematic car x position constant noise, which is normally distributed with standard deviation
KM_Y_FIX_NOISE = 0.01 # Kinematic car y position constant noise, which is normally distributed with standard deviation
KM_THETA_FIX_NOISE = 0.02 # Kinematic car theta constant noise, which is normally distributed with standard deviation
NUM_PARTICLES = 1000


'''
  Propagates the particles forward based on the velocity and steering angle of the car
'''
class KinematicMotionModel:
  '''
    Initializes the kinematic motion model
      motor_state_topic: The topic containing motor state information
      servo_state_topic: The topic containing servo state information    
      speed_to_erpm_offset: Offset conversion param from rpm to speed
      speed_to_erpm_gain: Gain conversion param from rpm to speed
      steering_angle_to_servo_offset: Offset conversion param from servo position to steering angle
      steering_angle_to_servo_gain: Gain conversion param from servo position to steering angle 
      car_length: The length of the car
      particles: The particles to propagate forward
      state_lock: Controls access to particles    
  '''
  def __init__(self, motor_state_topic, servo_state_topic, 
                     speed_to_erpm_offset, speed_to_erpm_gain, 
                     steering_to_servo_offset, steering_to_servo_gain, 
                     car_length, particles, state_lock=None):
    print "[Motion Model] Initialization..."
    self.last_servo_cmd = None # The most recent servo command
    self.last_vesc_stamp = None # The time stamp from the previous vesc_msg
    self.particles = particles
    self.num_particles = particles.shape[0]
    self.SPEED_TO_ERPM_OFFSET = speed_to_erpm_offset # Offset conversion param from rpm to speed
    self.SPEED_TO_ERPM_GAIN   = speed_to_erpm_gain # Gain conversion param from rpm to speed  
    self.STEERING_TO_SERVO_OFFSET = steering_to_servo_offset # Offset conversion param from servo position to steering angle
    self.STEERING_TO_SERVO_GAIN = steering_to_servo_gain # Gain conversion param from servo position to steering angle
    self.CAR_LENGTH = car_length # The length of the car
    self.incr = np.zeros((self.num_particles, 3))
    self.theta = np.zeros((self.num_particles, 2))
    self.theta_lmt = np.zeros(self.num_particles)
    self.delta = np.zeros(self.num_particles)
    self.speed = np.zeros(self.num_particles)

    
    # This just ensures that two different threads are not changing the particles
    # array at the same time. You should not have to deal with this.
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
      
    # This subscriber just caches the most recent servo position command
    self.servo_pos_sub  = rospy.Subscriber(servo_state_topic, Float64, self.servo_callback, queue_size=1)
    # Subscribe to the state of the vesc
    self.motion_sub = rospy.Subscriber(motor_state_topic, VescStateStamped, self.motion_callback, queue_size=1)
    print "[Motion Model] Initialization complete"                                     


  '''
    Caches the most recent servo command
      msg: A std_msgs/Float64 message
  '''
  def servo_callback(self, msg):
    self.last_servo_cmd = msg.data # Update servo command


  '''
    Converts messages to controls and applies the kinematic car model to the
    particles
      msg: a vesc_msgs/VescStateStamped message
  '''
  def motion_callback(self, msg):
    # print "[Motion Model] Future pose estimation..."
    self.state_lock.acquire()
    if self.last_servo_cmd is None:
      self.state_lock.release()
      return

    if self.last_vesc_stamp is None:
      self.last_vesc_stamp = msg.header.stamp
      self.state_lock.release()
      return
    
    # Convert raw msgs to controls
    # Note that control_val = (raw_msg_val - offset_param) / gain_param
    self.speed[:] = Gauss((msg.state.speed - self.SPEED_TO_ERPM_OFFSET) / self.SPEED_TO_ERPM_GAIN, KM_V_NOISE, self.num_particles)
    self.delta[:] = Gauss((self.last_servo_cmd - self.STEERING_TO_SERVO_OFFSET) / self.STEERING_TO_SERVO_GAIN, KM_DELTA_NOISE, self.num_particles)
    beta = np.arctan(np.tan(self.delta) / 2)
    # Propagate particles forward in place
      # Sample control noise and add to nominal control
      # Make sure different control noise is sampled for each particle
      # Propagate particles through kinematic model with noisy controls
      # Sample model noise for each particle
      # Limit particle theta to be between -pi and pi
      # Vectorize your computations as much as possible
      # All updates to self.particles should be in-place
    self.theta[:,0] = self.particles[:,2]
    dt = msg.header.stamp.to_sec() - self.last_vesc_stamp.to_sec()
    self.theta_lmt[:] = self.theta[:,0] + self.speed / self.CAR_LENGTH * np.sin(2 * beta) * dt
    self.last_vesc_stamp = msg.header.stamp
    for idx, theta_idx in enumerate(self.theta_lmt):
      if np.abs(theta_idx) > np.pi:
        self.theta_lmt[idx] = -np.sign(theta_idx) * (2*np.pi - np.abs(theta_idx))
    self.theta[:,1] = self.theta_lmt[:]

    self.incr[:,0] = Gauss(self.particles[:,0] + self.CAR_LENGTH / np.sin(2 * beta) * (np.sin(self.theta[:,1]) - np.sin(self.theta[:,0])), KM_X_FIX_NOISE)
    self.incr[:,1] = Gauss(self.particles[:,1] + self.CAR_LENGTH / np.sin(2 * beta) * (-np.cos(self.theta[:,1]) + np.cos(self.theta[:,0])), KM_Y_FIX_NOISE)
    self.incr[:,2] = Gauss(self.theta[:,1], KM_THETA_FIX_NOISE) 

    self.particles[:,:] = self.incr[:,:]

    self.last_vesc_stamp = msg.header.stamp    
    self.state_lock.release()
    # print "[Motion Model] Future poses estimated" 


'''
  Code for testing motion model
'''
TEST_SPEED = 1.5 # meters/sec
TEST_STEERING_ANGLE = 0.00 # radians
TEST_DT = 1.0 # seconds

if __name__ == '__main__':
  rospy.init_node("motion_model", anonymous=True) # Initialize the node
  particles = np.zeros((NUM_PARTICLES, 3))

  # Load params
  motor_state_topic = rospy.get_param("~motor_state_topic", "/vesc/sensors/core") # The topic containing motor state information
  servo_state_topic = rospy.get_param("~servo_state_topic", "/vesc/sensors/servo_position_command") # The topic containing servo state information
  speed_to_erpm_offset = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0)) # Offset conversion param from rpm to speed
  speed_to_erpm_gain = float(rospy.get_param("/vesc/speed_to_erpm_gain", 4350))   # Gain conversion param from rpm to speed, Default Given in the /home/robot/catkin_ws/src/mushr/mushr_base/vesc/vesc_main/config/racecar-uw-nano/vesc.yaml = -2000
  steering_angle_to_servo_offset = float(rospy.get_param("/vesc/steering_angle_to_servo_offset", 0.5)) # Offset conversion param from servo position to steering angle, Default Given in the vesc.yaml = 0.55
  steering_angle_to_servo_gain = float(rospy.get_param("/vesc/steering_angle_to_servo_gain", -1.2135)) # Gain conversion param from servo position to steering, Default Given in the vesc.yaml = 1.2135
  car_length = float(rospy.get_param("/car_kinematics/car_length", 0.33)) # The length of the car
    
  # Going to fake publish controls
  servo_pub = rospy.Publisher(servo_state_topic, Float64, queue_size=1)
  vesc_state_pub = rospy.Publisher(motor_state_topic, VescStateStamped, queue_size=1)
  
  KMM = KinematicMotionModel(motor_state_topic, servo_state_topic, speed_to_erpm_offset, 
                             speed_to_erpm_gain, steering_angle_to_servo_offset,
                             steering_angle_to_servo_gain, car_length, particles)
  
  # Give 1 sec to get setup
  rospy.sleep(1.0)
  
  # Send initial position and vesc state  
  servo_msg = Float64()
  servo_msg.data = steering_angle_to_servo_gain * TEST_STEERING_ANGLE + steering_angle_to_servo_offset
  
  servo_pub.publish(servo_msg)
  rospy.sleep(1.0)
  
  vesc_msg = VescStateStamped()
  vesc_msg.header.stamp = rospy.Time.now()
  vesc_msg.state.speed = speed_to_erpm_gain * TEST_SPEED + speed_to_erpm_offset  
  vesc_state_pub.publish(vesc_msg)
  
  rospy.sleep(TEST_DT)
  
  vesc_msg.header.stamp = rospy.Time.now()
  vesc_state_pub.publish(vesc_msg)
  
  rospy.sleep(1.0)
  
  KMM.state_lock.acquire()
  # Visualize particles
  plt.xlabel('x')
  plt.ylabel('y')
  plt_label = "Motion Model Parameters:\n Speed: %f m/s\n Steering Angle: %f rad\n dt: %f sec" % (TEST_SPEED, TEST_STEERING_ANGLE, TEST_DT)
  plt.scatter([0],[0], c='r', label=plt_label)
  plt_label = "Noise Parameters:\n Speed: %f\n Steering Angle: %f\n Posititon [x,y,theta]: [%f, %f, %f]" % (KM_V_NOISE, KM_DELTA_NOISE, KM_X_FIX_NOISE, KM_Y_FIX_NOISE, KM_THETA_FIX_NOISE)
  plt.scatter(particles[:,0], particles[:,1], c='b', label=plt_label)
  plt.legend()
  print "[Motion Model] Particles visualized" 
  plt.show()
  KMM.state_lock.release()