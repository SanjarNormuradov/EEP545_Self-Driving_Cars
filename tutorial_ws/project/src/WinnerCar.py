#!/usr/bin/env python

import rospy, sys, Utils
import numpy as np
from threading import Lock

from project.srv import *

from PlannerNode import PlannerNode
from ParticleFilter import ParticleFilter
from LineFollower import LineFollower
from MPController import MPController


class WinnerCar(object):

  def __init__(self, map_service_name, halton_points, disc_radius, collision_delta, source_topic, target_topic, plan_topic, service_topic, car_width, car_length, waypoint_topic,
                     motor_state_topic, servo_state_topic, speed_to_erpm_offset, speed_to_erpm_gain, steering_angle_to_servo_offset, steering_angle_to_servo_gain,
                     scan_topic, pose_topic, laser_ray_step, max_range_meters, Z_HIT, Z_RAND, Z_MAX, Z_SHORT, SIGMA_HIT, LAMBDA_SHORT, resample_type, n_particles, n_viz_particles, 
                     plan_lookahead, translation_weight, rotation_weight, Kp, Ki, Kd, error_buff_length, speed,
                     min_speed, max_speed, min_delta, max_delta, traj_nums, dt, T, compute_time, visual_sim=False, new_plan=False):

    print "[Winner Car] Planner node initialization..."
    self.planner_node = PlannerNode(map_service_name, halton_points, disc_radius, collision_delta, 
                                    source_topic, target_topic, plan_topic, service_topic, car_width, car_length, waypoint_topic)
    print "[Winner Car] Planner node initialization complete"

    print "[Winner Car] Waiting for service: " + service_topic
    rospy.wait_for_service(service_topic)
    get_plan = rospy.ServiceProxy(service_topic, GetPlan)
    print "[Winner Car] ...got service"
    try:
      plan_status = 'new' if new_plan else 'saved'
      print "[Winner Car] Requesting a %s plan..." % plan_status  
      reqst = get_plan(plan_status)
      if reqst.success:
        print "[Winner Car] Plan successfuly created!"
        plan = list(np.array(reqst.plan).reshape(-1, 3))
        self.planner_node.publish_plan(plan)
        raw_input("[Winner Car] Have a plan. Press Enter to start particle filter initialization") 
        print "[Winner Car] Particle filter initialization..."
        self.particle_filter = ParticleFilter(n_particles, n_viz_particles, resample_type, car_length, map_service_name,
                                              motor_state_topic, servo_state_topic, speed_to_erpm_offset, speed_to_erpm_gain, steering_angle_to_servo_offset, steering_angle_to_servo_gain, 
                                              scan_topic, pose_topic, laser_ray_step, max_range_meters, Z_HIT, Z_RAND, Z_MAX, Z_SHORT, SIGMA_HIT, LAMBDA_SHORT)
        print "[Winner Car] Particle filter initialization complete"
        raw_input("[Winner Car] Set initial pose if it is not set automatically. Then, press Enter")
        # while True:
        #   controller_type = input("[Winner Car] Which controller would you prefer?\nPID (1)\nMPC (2)\n")
        #   print "Controller Type: %s" % str(controller_type)
        #   if (int(controller_type) == 1) or (controller_type == 'PID' or 'pid'):
        #     print "[Winner Car] PID controller initialization..."
        #     self.pid_controller = LineFollower(plan, pose_topic, plan_lookahead, translation_weight,
        #                                        rotation_weight, Kp, Ki, Kd, error_buff_length, speed, map_service_name)
        #     print "[Winner Car] PID controller initialization complete. Plan execution..."
        #     break
        #   elif (int(controller_type) == 2) or (controller_type == 'MPC' or 'mpc'):
        print "[Winner Car] MPC controller initialization..."
        self.mpc_controller = MPController(plan, pose_topic, min_speed, max_speed, min_delta, max_delta, 
                                           traj_nums, dt, T, compute_time, car_length, car_width, visual_sim)
        print "[Winner Car] MPC controller initialization complete. Plan execution..."
          #   break
          # else:
          #   print "[Winner Car] Unrecognized controller type %s. Input valid type, please..." % str(controller_type)
      else:
        print "[Winner Car] Could not compute a plan..."
    except rospy.ServiceException, service_error:
      print "Service call failed: %s" % service_error                  


if __name__ == '__main__':
  rospy.init_node('winner_car', anonymous=True)
  
  # Planner Node Parameters
  map_service_name = rospy.get_param("~static_map", "static_map")
  halton_points = rospy.get_param("~halton_points", 1250)
  disc_radius = rospy.get_param("~disc_radius", 3)
  collision_delta = rospy.get_param("~collision_delta", 0.15)  
  source_topic = rospy.get_param("~source_topic" , "/initialpose")
  target_topic = rospy.get_param("~target_topic", "/move_base_simple/goal")
  plan_topic = rospy.get_param("~plan_topic", None)
  service_topic = rospy.get_param("~service_topic", None)
  car_width = rospy.get_param("/car_kinematics/car_width", 0.33)
  car_length = rospy.get_param("/car_kinematics/car_length", 0.33)
  waypoint_topic = rospy.get_param("~waypoint_topic", "/waypoint")

  # Motion Model Parameters
  speed_to_erpm_offset = float(rospy.get_param("/car/vesc/speed_to_erpm_offset", 0.0)) # Offset conversion param from rpm to speed
  speed_to_erpm_gain = float(rospy.get_param("/car/vesc/speed_to_erpm_gain", 4350))   # Gain conversion param from rpm to speed
  steering_angle_to_servo_offset = float(rospy.get_param("/car/vesc/steering_angle_to_servo_offset", 0.5)) # Offset conversion param from servo position to steering angle
  steering_angle_to_servo_gain = float(rospy.get_param("/car/vesc/steering_angle_to_servo_gain", -1.2135)) # Gain conversion param from servo position to steering angle
  motor_state_topic = rospy.get_param("~motor_state_topic", "/car/vesc/sensors/core")
  servo_state_topic = rospy.get_param("~servo_state_topic", "/car/vesc/sensors/servo_position_command")

  # Sensor Model Parameters
  scan_topic = rospy.get_param("~scan_topic", "/car/scan")
  pose_topic = rospy.get_param("~pose_topic", "/pf/vis/inferred_pose")
  laser_ray_step = rospy.get_param("~laser_ray_step", 20)
  max_range_meters = rospy.get_param("~max_range_meters", 5.6)
  Z_HIT = rospy.get_param("~Z_HIT", 0.55)
  Z_RAND = rospy.get_param("~Z_RAND", 0.05)
  Z_MAX = rospy.get_param("~Z_MAX", 0.25)
  Z_SHORT = rospy.get_param("~Z_SHORT", 0.15)
  SIGMA_HIT = rospy.get_param("~SIGMA_HIT", 7.5)
  LAMBDA_SHORT = rospy.get_param("~LAMBDA_SHORT", 0.001)

  # ReSampler Parameters
  resample_type = rospy.get_param("~resample_type", "low_variance")

  # Particle Filter Parameters
  n_particles = rospy.get_param("~n_particles", 1000)
  n_viz_particles = rospy.get_param("~n_viz_particles", 20)
  # motion_model = rospy.get_param("~motion_model", "kinematic")
  # odometry_topic = rospy.get_param("~odometry_topic", "/vesc/odom")

  # PID Controller Parameters
  plan_lookahead = rospy.get_param("~plan_lookahead", 7)
  translation_weight = rospy.get_param("~translation_weight", 1.0)
  rotation_weight = rospy.get_param("~rotation_weight", 0.2)
  Kp = rospy.get_param("~Kp", 1.0)
  Ki = rospy.get_param("~Ki", 0.1)
  Kd = rospy.get_param("~Kd", 0.0)
  error_buff_length = rospy.get_param("~error_buff_length", 10)
  speed = rospy.get_param("~speed", 1.0)

  # MPC Controller Parameters
  compute_time = rospy.get_param("~compute_time", 0.09)
  min_delta = rospy.get_param("~min_delta", -0.34)
  max_delta = rospy.get_param("~max_delta", 0.341)
  traj_nums = rospy.get_param("~traj_nums", 9)
  dt = rospy.get_param("~dt", 0.03)
  T = rospy.get_param("~T", 270)
  max_speed = rospy.get_param("~max_speed", 2.0)
  min_speed = rospy.get_param("~min_speed", 1.0)


  WC = WinnerCar(map_service_name, halton_points, disc_radius, collision_delta, source_topic, target_topic, plan_topic, service_topic, car_width, car_length, waypoint_topic,
                 motor_state_topic, servo_state_topic, speed_to_erpm_offset, speed_to_erpm_gain, steering_angle_to_servo_offset, steering_angle_to_servo_gain,
                 scan_topic, pose_topic, laser_ray_step, max_range_meters, Z_HIT, Z_RAND, Z_MAX, Z_SHORT, SIGMA_HIT, LAMBDA_SHORT, resample_type, n_particles, n_viz_particles, 
                 plan_lookahead, translation_weight, rotation_weight, Kp, Ki, Kd, error_buff_length, speed,
                 min_speed, max_speed, min_delta, max_delta, traj_nums, dt, T, compute_time, visual_sim=True, new_plan=False)
                   
  while not rospy.is_shutdown():
    # Callbacks are running in separate threads
    if WC.particle_filter.sensor_model.do_resample: # Check if the sensor model says it's time to resample
      WC.particle_filter.sensor_model.do_resample = False # Reset so that we don't keep resampling
      
      # Resample
      if WC.particle_filter.RESAMPLE_TYPE == "naiive":
        WC.particle_filter.resampler.resample_naiive()
      elif WC.particle_filter.RESAMPLE_TYPE == "low_variance":
        WC.particle_filter.resampler.resample_low_variance()
      else:
        print "[Particle Filter] Unrecognized resampling method: " + WC.particle_filter.RESAMPLE_TYPE      
      
      WC.particle_filter.visualize() # Perform visualization
