#!/usr/bin/env python

import rospy
import numpy as np
from lab5.srv import *
import Utils
from nav_msgs.srv import GetMap

PLANNER_SERVICE_TOPIC = '/planner_node/get_car_plan'

# Testing pose sets:
SOURCE1 = [156, 1080, 0.0]
TARGET1 = [519, 828, 0.0]

SOURCE2 = [765, 504, 0.0]
TARGET2 = [1608, 729, 0.0]

SOURCE3 = [2328, 462, 0.0]
TARGET3 = [456, 732, 0.0]

if __name__ == '__main__':

    rospy.init_node('planner_test', anonymous=True)

    map_service_name = rospy.get_param("~static_map", "static_map")
    print "[Planner Test] Getting map from service: " + map_service_name
    rospy.wait_for_service(map_service_name)
    map_info = rospy.ServiceProxy(map_service_name, GetMap)().map.info
    print "[Planner Test] ...got map"

    print "[Planner Test] Waiting for service: " + PLANNER_SERVICE_TOPIC
    rospy.wait_for_service(PLANNER_SERVICE_TOPIC)
    get_plan = rospy.ServiceProxy(PLANNER_SERVICE_TOPIC, GetPlan)

    try:
        print "[Planner Test] Requesting a plan..."
        resp = get_plan(Utils.map_to_world(SOURCE1, map_info), Utils.map_to_world(TARGET1, map_info))
        if resp.success:
            print "[Planner Test] Plan successfuly created: \n%s" % np.array(resp.plan).reshape(-1, 3)
        else:
            print "[Planner Test] Could not compute a plan..."
    except rospy.ServiceException, e:
        print 'Service call failed: %s' % e

