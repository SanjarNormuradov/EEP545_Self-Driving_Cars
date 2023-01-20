#!/usr/bin/env python

import rospy, sys, csv
import numpy as np
from threading import Lock

from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from project.srv import *

from HaltonPlanner import HaltonPlanner
from HaltonEnvironment import HaltonEnvironment
import GraphGenerator, Utils

PUB_RATE = 30


class PlannerNode(object):

  def __init__(self, map_service_name, 
                     halton_points, 
                     disc_radius,
                     collision_delta,                      
                     source_topic,
                     target_topic,
                     plan_topic,
                     service_topic,
                     car_width,
                     car_length,
                     waypoint_topic):
    
    print "[Planner Node] Initialization..."
    print "[Planner Node] Getting map from service..."
    rospy.wait_for_service(map_service_name)
    self.map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
    print "[Planner Node] ...got map"
    
    print "[Planner Node] Generating graph file..."
    # graph_file is just ".....graphml" which consist of empty Directed Graph template without vertices, because source and target are not given
    graph_file = GraphGenerator.generate_graph_file(self.map_msg, halton_points, disc_radius, car_width, car_length, collision_delta)
    print "[Planner Node] ..graph generated"
    
    rospy.Rate(PUB_RATE)
    # self.environment is the graph read from "graph_file", i.e. empty graph
    self.environment = HaltonEnvironment(self.map_msg, graph_file, None, None, car_width, car_length, disc_radius, collision_delta, waypoint_topic)
    self.planner = HaltonPlanner(self.environment)
    
    self.source_pose = None
    self.source_updated = False
    self.source_yaw = None
    self.source_lock = Lock()
    self.target_pose = None
    self.target_updated = False
    self.target_yaw = None
    self.target_lock = Lock()
    
    self.cur_plan = None
    self.plan_lock = Lock()
    
    self.orientation_window_size = 21
    self.is_plan_published = False

    if waypoint_topic is not None:
      self.waypoints_pub = rospy.Publisher(waypoint_topic,
                                           MarkerArray,
                                           queue_size=100)
    # Graph nodes (i.e. states) are real world coordinates (meters)
    bad_waypoints = []
    self.waypoints = []
    print "[Planner Node] Generating path waypoints..."
    with open(rospy.get_param("~startpoint_filename"), "r") as csv_file:
      waypoints_file = csv.DictReader(csv_file)
      for crd_dict in waypoints_file:
        self.waypoints.append(Utils.map_to_world_s([int(crd_dict['x']), int(crd_dict['y'])], self.map_msg.info))
    self.publish_waypoints(self.waypoints, wypnt_type='start')
    print "[Planner Node] Starting waypoint is visualized!"

    with open(rospy.get_param("~goodwaypoints_filename"), "r") as csv_file:
      waypoints_file = csv.DictReader(csv_file)
      for crd_dict in waypoints_file:
        self.waypoints.append(Utils.map_to_world_s([int(crd_dict['x']), int(crd_dict['y'])], self.map_msg.info))
    self.publish_waypoints(self.waypoints[1:], wypnt_type='good')
    print "[Planner Node] Good waypoints are visualized!"

    with open(rospy.get_param("~badwaypoints_filename"), "r") as csv_file:
      waypoints_file = csv.DictReader(csv_file)
      for crd_dict in waypoints_file:
        bad_waypoints.append(Utils.map_to_world_s([int(crd_dict['x']), int(crd_dict['y'])], self.map_msg.info))
    self.publish_waypoints(bad_waypoints, wypnt_type='bad')
    print "[Planner Node] Bad waypoints are visualized!"
    print "[Planner Node] Path waypoints are generated"
    
    # plan_topic = "/winner_car/car_plan"
    if plan_topic is not None:
      self.plan_pub = rospy.Publisher(plan_topic, 
                                      PoseArray, 
                                      queue_size=1000)  
      # self.source_sub = rospy.Subscriber(source_topic, 
      #                                    PoseWithCovarianceStamped, 
      #                                    self.source_callback,
      #                                    queue_size=100)
      # self.target_sub = rospy.Subscriber(target_topic, 
      #                                    PoseStamped, 
      #                                    self.target_callback,
      #                                    queue_size=100)          
    else:
      self.plan_pub = None
                                         
    if service_topic is not None:
      self.plan_service = rospy.Service(service_topic, GetPlan, self.get_plan_callback)
    else:
      self.plan_service = None
    
    print '[Planner Node] Initialization complete. Ready to plan'
    

  def get_plan_callback(self, reqst):
    print "[Planner Node] Service request"
    print "[Planner Node] Generating response..."
    resp = GetPlanResponse()
    subplan = []
    if reqst.plan_status == 'new':
      for waypoint in self.waypoints:
        if self.source_pose is None:
          # print "[Planner Node] 1st source is set"
          self.source_lock.acquire()
          self.source_pose = waypoint
          self.source_yaw = 0.0
          self.source_updated = True
          self.source_lock.release()
        elif self.target_pose is None:
          # print "[Planner Node] 1st target is set"
          self.target_lock.acquire()
          self.target_pose = waypoint
          self.target_yaw = 0.0
          self.target_updated = True
          self.target_lock.release()
        else:
          # print "[Planner Node] Reached target is set as new source"
          self.source_lock.acquire()
          self.source_pose[:] = self.target_pose[:]
          self.source_yaw = self.target_yaw
          self.source_updated = True
          self.source_lock.release()
          # print "[Planner Node] New target is set"
          self.target_lock.acquire()
          self.target_pose = waypoint
          self.target_yaw = 0.0
          self.target_updated = True
          self.target_lock.release()

        if (self.source_pose is not None) and (self.target_pose is not None):
          self.is_plan_simple = False
          self.plan_lock.acquire()
          self.update_plan()
          self.plan_lock.release()        
          resp.success = True if self.cur_plan is not None else False
          if resp.success:
            # self.cur_plan is flat list of [x,y,theta]
            subplan.append(self.cur_plan)
      resp.success = True
      # subplan = LIST(List1, List2, ... ListN), where N is len(self.waypoints)-1 and List = list([x1,y1,theta1], [x2,y2,theta2]....[xn,yn,thetan]). 
      # [x1,y1,theta1] is coordinates of source, [xn,yn,thetan] is coordinates of target
      flat_subplan = [item for sublist in subplan for item in sublist]
      # Plan has unoriented nodes 
      subplan = self.add_orientation(flat_subplan)
      # Now subplan is np.array(N, 3)
      # resp.plan should be plain list of float32
      resp.plan = [crdnt for crdnts in subplan for crdnt in crdnts]
      fields = ['x', 'y', 'theta']
      with open(rospy.get_param("~plan_filename"), "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        csv_writer.writerows(np.array(resp.plan).reshape(-1, 3))

    elif reqst.plan_status == 'saved':
      resp.success = True
      subplan = []
      with open(rospy.get_param("~plan_filename"), "r") as csv_file:
        plan_file = csv.DictReader(csv_file)
        for crd_dict in plan_file:
          subplan.append([float(crd_dict['x']), float(crd_dict['y']), float(crd_dict['theta'])])
      resp.plan = [crdnt for crdnts in subplan for crdnt in crdnts]

    print "[Planner Node] Response generated!"
    return resp 
    

  # def source_callback(self, msg):
  #   self.source_lock.acquire()
  #   self.source_pose = [msg.pose.pose.position.x,
  #                       msg.pose.pose.position.y]
  #   self.source_yaw = Utils.quaternion_to_angle(msg.pose.pose.orientation)
  #   self.source_updated = True
  #   print '[Planner Node] Got new source [x,y,theta]: [%f, %f, %f]' %(self.source_pose[0], self.source_pose[1], self.source_yaw)
  #   self.source_lock.release()
    

  # def target_callback(self, msg):  
  #   self.target_lock.acquire()
  #   self.target_pose = [msg.pose.position.x,
  #                       msg.pose.position.y]
  #   self.target_yaw = Utils.quaternion_to_angle(msg.pose.orientation)
  #   self.target_updated = True
  #   print '[Planner Node] Got new target [x,y,theta]: [%f, %f, %f]' %(self.target_pose[0], self.target_pose[1], self.target_yaw)
  #   self.is_plan_published = False
  #   self.is_plan_simple = True
  #   self.target_lock.release()
    

  def publish_plan(self, plan):
    pa = PoseArray()
    pa.header = Utils.make_header("/map")
    for i in xrange(len(plan)):
      config = plan[i]
      pose = Pose()
      pose.position.x = config[0]
      pose.position.y = config[1]
      pose.position.z = 0.0
      pose.orientation = Utils.angle_to_quaternion(config[2])
      pa.poses.append(pose)
    rospy.sleep(1.0)
    self.plan_pub.publish(pa)


  def publish_waypoints(self, waypoints, wypnt_type):
    ma = MarkerArray()
    for i in xrange(len(waypoints)):
      waypoint = waypoints[i]
      marker = Marker()
      marker.header = Utils.make_header("/map")
      marker.type = marker.CUBE
      marker.action = marker.ADD
      marker.pose.position.x = waypoint[0]
      marker.pose.position.y = waypoint[1]
      marker.pose.position.z = 0.0
      marker.pose.orientation = Utils.angle_to_quaternion(0.0)
      marker.scale.x = 0.33
      marker.scale.y = 0.33
      marker.scale.z = 0.03

      if wypnt_type == 'start':
        marker.ns = 'start_waypoint'
        marker.id = 1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

      elif wypnt_type == 'good':
        marker.ns = 'good_waypoints'
        marker.id = i + 1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

      elif wypnt_type == 'bad':
        marker.ns = 'bad_waypoints'
        marker.id = i + 1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
      ma.markers.append(marker)
    rospy.sleep(0.5)
    self.waypoints_pub.publish(ma) 


  def add_orientation(self, plan):
    plan = np.array(plan)

    oriented_plan = np.zeros((plan.shape[0], 3), dtype=np.float)
    oriented_plan[:, 0:2] = plan[:,:]
 
    if plan.shape[0] >= 2:  
      oriented_plan[0,2] = self.source_yaw
      oriented_plan[oriented_plan.shape[0]-1, 2] = self.target_yaw   
          
      plan_diffs = np.zeros(plan.shape, np.float)
      plan_diffs[0:plan_diffs.shape[0]-1] = plan[1:plan.shape[0]] - plan[0:plan.shape[0]-1]
      plan_diffs[plan_diffs.shape[0]-1] = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw)], dtype=np.float) 
    
      avg_diffs = np.empty(plan_diffs.shape, dtype=np.float)
      for i in xrange(plan_diffs.shape[0]):
        avg_diffs[i] = np.mean(plan_diffs[np.max((0, i-self.orientation_window_size/2)):
                                          np.min((plan_diffs.shape[0]-1, i+self.orientation_window_size/2+1))], axis=0)

      oriented_plan[1:oriented_plan.shape[0]-1, 2] = np.arctan2(avg_diffs[1:oriented_plan.shape[0]-1, 1],
                                                               avg_diffs[1:oriented_plan.shape[0]-1, 0])
   
    elif plan.shape[0] == 2:
      oriented_plan[:,2] = np.arctan2(plan[1,1]-plan[0,1], plan[1,0]-plan[0,0])

    return oriented_plan
      

  def update_plan(self):
    self.source_lock.acquire()
    self.target_lock.acquire()
    if self.source_pose is not None:
      source_pose = np.array(self.source_pose).reshape(2)
    if self.target_pose is not None:
      target_pose = np.array(self.target_pose).reshape(2)
    replan = ((self.source_updated or self.target_updated) and
              (self.source_pose is not None and self.target_pose is not None))
    self.source_updated = False
    self.target_updated = False
    self.source_lock.release()
    self.target_lock.release()
    
    if replan:
      if(np.abs(source_pose-target_pose).sum() < sys.float_info.epsilon):
        print '[Planner Node] Source and target are the same, will not plan'
        return

      if not self.environment.manager.get_state_validity(source_pose):
        print '[Planner Node] Source in collision, will not plan'
        return

      if not self.environment.manager.get_state_validity(target_pose):
        print '[Planner Node] Target in collision, will not plan'
        return

      print '[Planner Node] Inserting source and target'
      self.environment.set_source_and_target(source_pose, target_pose)

      print '[Planner Node] Computing plan...'      
      self.cur_plan = self.planner.plan()    
      
      if self.cur_plan is not None:
        # print "[Planner Node] Path length before post_processing: %f" %self.planner.get_path_length(self.cur_plan)
        # self.cur_plan = self.planner.post_process(self.cur_plan, 5)
        # print "[Planner Node] Path length after post_processing: %f\n" %self.planner.get_path_length(self.cur_plan)
        # self.cur_plan = self.add_orientation(self.cur_plan)
        if self.is_plan_simple:
          # self.planner.visualize(self.cur_plan)
          print '[Planner Node] ...plan complete'
      else:
        print '[Planner Node] ...could not compute a plan'
      
    if (self.cur_plan is not None) and (self.plan_pub is not None) and not self.is_plan_published and self.is_plan_simple:
      self.publish_plan(self.cur_plan)  
      self.is_plan_published = True
      self.source_pose = None
      self.target_pose = None
