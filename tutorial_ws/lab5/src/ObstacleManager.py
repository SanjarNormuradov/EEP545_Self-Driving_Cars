import cv2, numpy, math, Utils


class ObstacleManager(object):

	def __init__(self, mapMsg, 
					   car_width, 
					   car_length, 
					   collision_delta):
		print "[Obstacle Manager] Initialization..."
		self.map_info = mapMsg.info
		self.mapImageGS = numpy.array(mapMsg.data, dtype=numpy.uint8).reshape(
			(mapMsg.info.height, mapMsg.info.width, 1))

		# Retrieve the map dimensions
		self.mapHeight, self.mapWidth, self.mapChannels = self.mapImageGS.shape
		print "[Obstacle Manager] Map: Height: %d, Width: %d, Channels: %d" %(self.mapHeight, self.mapWidth, self.mapChannels)

		# Binarize the Image
		self.mapImageBW = 255 * numpy.ones_like(self.mapImageGS, dtype=numpy.uint8)
		self.mapImageBW[self.mapImageGS == 0] = 0

		# Obtain the car length and width in pixels
		self.robotHalfWidth = int(math.ceil(int(car_width / self.map_info.resolution + 0.5) / 2))
		self.robotHalfLength = int(math.ceil(int(car_length / self.map_info.resolution + 0.5) / 2))
		self.collision_delta = collision_delta


	# Check if the passed config is in collision
	# config: The configuration to check (in meters and radians)
	# Returns False if in collision, True if not in collision
	def get_state_validity(self, config):
		# print "[Obstacle Manager] State validation..."
		# Convert the configuration to map-coordinates -> mapConfig is in pixel-space [(0, self.Width), (0, self.Height)]
		mapConfig = Utils.world_to_map(config, self.map_info)

		# Return true or false based on whether the robot's configuration is in collision
		# Use a square to represent the robot, return true only when all points within
		# the square are collision free
		#
		# Also return false if the robot is out of bounds of the map
		#
		# Although our configuration includes rotation, assume that the
		# square representing the robot is always aligned with the coordinate axes of the
		# map for simplicity
		if ((mapConfig[1] + self.robotHalfLength > self.mapHeight - 1) or
			(mapConfig[1] - self.robotHalfLength < 0) or
			(mapConfig[0] + self.robotHalfWidth > self.mapWidth - 1) or
			(mapConfig[0] - self.robotHalfWidth < 0)):
			# print "[Obstacle Manager] Car will be out of map boundaries!"
			return False
		# print "[Obstacle Manager] Car is still within map boundaries..."
		pnt_w = self.robotHalfWidth
		for pnt_l in range(1, self.robotHalfLength + 1):
			if (((self.mapImageBW[mapConfig[1] + pnt_l, mapConfig[0] + pnt_w]) +
				 (self.mapImageBW[mapConfig[1] - pnt_l, mapConfig[0] + pnt_w]) +
				 (self.mapImageBW[mapConfig[1] + pnt_l, mapConfig[0] - pnt_w]) +
				 (self.mapImageBW[mapConfig[1] - pnt_l, mapConfig[0] - pnt_w])) != 0):
				#  print "[Obstacle Manager] Car will hit an obstacle!"
				 return False

		pnt_l = self.robotHalfLength
		for pnt_w in range(1, self.robotHalfWidth + 1):
			if (((self.mapImageBW[mapConfig[1] + pnt_l, mapConfig[0] + pnt_w]) +
				 (self.mapImageBW[mapConfig[1] - pnt_l, mapConfig[0] + pnt_w]) +
				 (self.mapImageBW[mapConfig[1] + pnt_l, mapConfig[0] - pnt_w]) +
				 (self.mapImageBW[mapConfig[1] - pnt_l, mapConfig[0] - pnt_w])) != 0):
				#  print "[Obstacle Manager] Car will hit an obstacle!"
				 return False
		# print "[Obstacle Manager] Car can move to this point!"
		return True


	# Discretize the path into N configurations, where N = path_length / self.collision_delta
	#
	# input: an edge represented by the start and end configurations
	#
	# return three variables:
	# list_x - a list of x values of all intermediate points in the path
	# list_y - a list of y values of all intermediate points in the path
	# edgeLength - The euclidean distance between config1 and config2
	def discretize_edge(self, config1, config2):
		# print "[Obstacle Manager] Edge discretization..."
		config1 = numpy.array(config1)
		config2 = numpy.array(config2)
		list_x, list_y = [], []
		edgeLength = 0
		edgeLength = numpy.sqrt(numpy.power(config1[:2] - config2[:2], 2).sum())
		N = int(math.ceil(edgeLength / self.collision_delta))
		for n in range(N):
			x, y = config1 + n * (config2 - config1) / N
			list_x.append(x)
			list_y.append(y)
		# print "[Obstacle Manager] Edge is discretazed into %d parts" %N
		return list_x, list_y, edgeLength


	# Check if there is an unobstructed edge between the passed configs
	# config1, config2: The configurations to check (in meters and radians)
	# Returns false if obstructed edge, True otherwise
	def get_edge_validity(self, config1, config2):
		# print "[Obstacle Manager] Validating edge between two nodes..."
		# Check if endpoints are obstructed, if either is, return false
		# Find path between two configs by connecting them with a straight line
		# Discretize the path with the discretized_edge function above
		# Check if all configurations along path are obstructed
		if self.get_state_validity([config2[0], config2[1], 0.0]) == False:
			# print "[Obstacle Manager] Target node is obstructed!"
			return False
		list_x, list_y, edgeLength = self.discretize_edge(config1, config2)
		for x, y in zip(list_x, list_y):
			if self.get_state_validity([x, y, 0.0]) == False:
				# print "[Obstacle Manager] One of discretized edges is obstructed"
				return False
		# print "[Obstacle Manager] Edge is not obstructed!"
		return True

	def update_path(self, config1, config2):
		# print "[Obstacle Manager] Path update estimation..."
		path_x, path_y, path_length = self.discretize_edge(config1, config2)
		valid = True
		if len(path_x) < 2:
			valid = False
			# print "[Obstacle Manager] Path is too short"
		else:
			for x, y in zip(path_x, path_y):
				if not self.get_state_validity([x, y, 0.0]):
					# print "[Obstacle Manager] Path segment is obstructed!"
					valid = False
					break
		if valid:
			# print "[Obstacle Manager] Path can be updated!"
			return True, path_x, path_y
		return False, None, None


# if __name__ == '__main__':
# 	return
