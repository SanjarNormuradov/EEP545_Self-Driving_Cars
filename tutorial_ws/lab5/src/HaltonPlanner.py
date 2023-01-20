import rospy, math, numpy, cv2, Utils, time, random, HaltonEnvironment
from matplotlib import pyplot as plt
from heapq import *

class HaltonPlanner(object):
  
  # planningEnv: Should be a HaltonEnvironment
  def __init__(self, planningEnv):
    self.planningEnv = planningEnv

  # Generate a plan
  # Assumes that the source and target were inserted just prior to calling this
  # Returns the generated plan
  def plan(self):
    planStart = time.time()
    print "[Halton Planner] Plan generation..."
    self.sid = self.planningEnv.graph.number_of_nodes() - 2 # Get source id
    self.tid = self.planningEnv.graph.number_of_nodes() - 1 # Get target id

    self.closed = set() # The closed list
    self.parent = {self.sid: None} # A dictionary mapping children to their parents
    self.open = list() # The open list
    heappush(self.open, (0 + self.planningEnv.get_heuristic(self.sid, self.tid), self.sid))
    self.gValues = {self.sid: 0} # A mapping from node to shortest found path length to that node 
    self.planIndices = []
    self.cost = 0
    target_is_achieved = False
    countNode = 0
    countNodeOpen = 0
    countNodeClosed = 0

    # Implement A*
    # Functions that you will probably use
    # - self.get_solution()
    # - self.planningEnv.get_successors()
    # - self.planningEnv.get_distance()
    # - self.planningEnv.get_heuristic()
    # Note that each node in the graph has both an associated id and configuration
    # You should be searching over ids, not configurations. get_successors() will return
    #   the ids of nodes that can be reached. Once you have a path plan
    #   of node ids, get_solution() will compute the actual path in SE(2) based off of
    #   the node ids that you have found.
    while self.open:
      currentNodeF, currentNodeID = heappop(self.open)
      self.closed.add(currentNodeID)

      if currentNodeID == self.tid:
        print "[Halton Planner] Target is achieved! Unoptimized distance from source: %f" %currentNodeF
        target_is_achieved = True 
        break

      for childNodeID in self.planningEnv.get_successors(currentNodeID):
        if childNodeID in self.closed: continue

        # Child node in collision, no need to account
        if not self.planningEnv.manager.get_state_validity(self.planningEnv.get_config(childNodeID)):
          # print "[Halton Planner] Child node is obstructed!"
          countNodeClosed += 1
          self.closed.add(childNodeID)
          continue
        # Edge from current parent node to child node is obstructed, not likely because they are not neighbouring nodes (points in the map) because of Halton Sequence generation
        if not self.planningEnv.manager.get_edge_validity(self.planningEnv.get_config(currentNodeID), self.planningEnv.get_config(childNodeID)):
          # print "[Halton Planner] Edge between current node and child is obstructed!"
          continue 

        childNodeG_old = self.gValues.get(childNodeID, None)
        childNodeG_new = self.planningEnv.get_distance(currentNodeID, childNodeID) + self.gValues[currentNodeID]
        if childNodeG_old is None or childNodeG_new < childNodeG_old:
          self.parent[childNodeID] = currentNodeID
          self.gValues[childNodeID] = childNodeG_new
          childNodeH = self.planningEnv.get_heuristic(childNodeID, self.tid)
          childNodeF = childNodeG_new + childNodeH
          heappush(self.open, (childNodeF, childNodeID))

    countNodeOpen = len(self.gValues) + 1
    countNode = countNodeOpen + countNodeClosed
    print "[Halton Planner] Number of Nodes (source and goal included):"
    print "[Halton Planner] Available: %d; Obstructed: %d; Total: %d" %(countNodeOpen, countNodeClosed, countNode)
    pathPlan = self.get_solution(self.tid) if target_is_achieved else []
    planEnd = time.time()
    self.planTime = planEnd - planStart
    self.planLength = self.get_path_length(pathPlan)
    print "[Halton Planner] Planning time: %.3f" % self.planTime
    print
    return pathPlan


  # Try to improve the current plan by repeatedly checking if there is a shorter path between random pairs of points in the path
  def post_process(self, plan, timeout):
    print "[Halton Planner] Post processing..."
    t1 = time.time()
    elapsed = 0
    countUpdate = 0
    while elapsed < timeout: # Keep going until out of time
      # print "[Halton Planner] Plan: %s" %str(plan)
      # print
      # print "Timeout: %d, Elapsed: %f" %(timeout, elapsed)
      # print "[Halton Planner] Path length: %f, size: %d" %(self.get_path_length(plan), len(plan))
      # Pick random id i
      # Pick random id j
      # Redraw if i == j
      # Switch i and j if i > j
      i, j = 0, 0
      while i == j:
        randomPick = numpy.random.randint(0, len(plan) - 1, 2)
        i = randomPick.min()
        j = randomPick.max()
      # print "[Halton Planner] i: %d" %i
      # print "[Halton Planner] j: %d" %j
      ''' 
      i = random.randint(len(plan))
      j = random.randint(len(plan))
      If plan is list(x1, y1, x2, y2, ...), where x's index is even and y's is odd
      No of interest cases:
        i == j: obvious
        j - i == 1: i is x1, j is y1 (same node); if i > j, same result; Possibly i is y1, j is x2 (neighbouring nodes)
        j - i == 2: i is x1, j is x2 (neighbouring nodes); if i > j, same result
        j - i == 3 and j is odd: i is x1, j is y2 (neighbouring nodes)
        i - j == 3 and i is odd: j is x1, i is y2 (neighbouring nodes)
      while (math.abs(i - j) <= 2) or ((j % 2) * (j - i) + (i % 2) * (i - j) == 3):
         j = random.randint(len(plan))
      if i > j:
        i, j = j, i
      # i (or j) is even (or odd): iX is x1, iY is y1; Always (x,y) of the same node
      iX, iY =  (i - (i % 2)), (i - (i % 2) + 1)
      jX, jY =  (j - (j % 2)), (j - (j % 2) + 1)
      '''
      config1 = plan[i]
      config2 = plan[j]
      # print "[Halton Planner] config1: %s" %str(config1)
      # print "[Halton Planner] config2: %s" %str(config2)
      # if we can find path between i and j (Hint: look inside ObstacleManager.py for a suitable function)
        # Get the path
        # Reformat the plan such that the new path is inserted and the old section of the path is removed between i and j
        # Be sure to CAREFULLY inspect the data formats of both the original plan and the plan returned
        # to ensure that you edit the path correctly
      valid, path_x, path_y = self.planningEnv.manager.update_path(config1, config2)
      if valid:
        # print "[Halton Planner] There is shorter path!"
        countUpdate += 1
        subplan = [list(a) for a in zip(path_x, path_y)]
        plan[i:j+1] = subplan[:]
        # print "[Halton Planner] Plan is updated with shorter path! Path length: %f, size: %d" % (self.get_path_length(plan), len(plan))

      elapsed = time.time() - t1
    print "[Halton Planner] Post processing done!"
    print "[Halton Planner] Number of updates: %d" %countUpdate
    self.planLength = self.get_path_length(plan)
    return plan


  # Backtrack across parents in order to recover path
  # vid: The id of the last node in the graph
  def get_solution(self, vid):
    print "[Halton Planner] Getting plan..."
    # Get all the node ids
    planID = []
    while vid is not None:
      planID.append(vid)
      vid = self.parent[vid]

    plan = []
    planID.reverse()
    for pathNode in planID:
      pose = self.planningEnv.get_config(pathNode)
      print "[Halton Planner] Path node: %d, [X,Y]: [%f, %f]" %(pathNode, pose[0], pose[1])

    for i in range(len(planID) - 1):
      startConfig = self.planningEnv.get_config(planID[i])
      goalConfig = self.planningEnv.get_config(planID[i + 1])
      # print "[Halton Planner] Plan: adding path from %s to %s" %(startConfig, goalConfig)
      px, py, clen = self.planningEnv.manager.discretize_edge(startConfig, goalConfig)
      plan.append([list(a) for a in zip(px, py)])
      self.planIndices.append(len(plan))
      self.cost += clen

    # Plan is a list(N1 'list's), where N1 is number of nodes in the path - 1. 
    # Each 'list' is a list(N2 of [x,y]), where N2 is edgeLength/collision_delta, i.e number of discretized path's parts
    # So flatPlan is a list(N1*N2 [x,y])
    print "[Halton Planner] Got plan!"
    flatPlan = [item for sublist in plan for item in sublist]
    return flatPlan


  def get_path_length(self, plan):
    plan = numpy.array(plan)
    length = numpy.sqrt(numpy.power(plan[1:plan.shape[0],0:2] - plan[0:plan.shape[0]-1,0:2], 2).sum(1)).sum()
    return length

  # Visualize the plan
  def visualize(self, plan):
    # Get the map
    envMap = 255*(self.planningEnv.manager.mapImageBW+1) # Hacky way to get correct coloring
    envMap = cv2.cvtColor(envMap, cv2.COLOR_GRAY2RGB)
    
    for i in range(numpy.shape(plan)[0]-1): # Draw lines between each configuration in the plan
      startPixel = Utils.world_to_map(plan[i], self.planningEnv.manager.map_info)
      goalPixel = Utils.world_to_map(plan[i+1], self.planningEnv.manager.map_info)
      cv2.line(envMap,(startPixel[0],startPixel[1]),(goalPixel[0],goalPixel[1]),(255,0,0),5)

    # Generate window
    cv2.namedWindow('Simulation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Simulation', 1236, 2792)
    text1 = "Halton Points: " + str(rospy.get_param("~halton_points")) + "; Disc Radius: " + str(rospy.get_param("~disc_radius"))
    text2 = "Post Prosses: Yes; Planning Time: " + str(round(self.planTime, 2)) + "; Path Length: " + str(round(self.planLength, 2))
    cv2.putText(envMap, text1, (70, 1100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(envMap, text2, (70, 1150), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow('Simulation', envMap)
    cv2.imwrite("/home/robot/tutorial_ws/src/lab5/pictures/planner_test1_7.png", envMap)

    # Terminate and exit elegantly
    cv2.waitKey(20000)
    cv2.destroyAllWindows()
