import sys

PYTHON_API_DIR = "/media/govvijay/data/carla/carla_0_9_9/PythonAPI/"
try:
    sys.path.append(PYTHON_API_DIR + '/carla')
except IndexError:
    pass

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
import vg_trajectory_utils as vtu

import pdb

class MPCBehavior(object):
    """ This is based on the Normal Agent. """

    max_speed = 50               # (km/h); max speed this agent will drive 
    speed_lim_dist = 3           # (km/h); how much slower than the speed limit to drive, if speed limit constraint is active 
    speed_decrease = 10	         # (km/h); speed drop when under the TTC threshold 
    safety_time = 3              # (sec); TTC threshold, if we are under this we should slow down 
    min_proximity_threshold = 10 # (m); minimum distance a vehicle should be to consider a possible hazard, i.e. follow it or change lanes 
    braking_distance = 5         # (m); if we are this close to a pedestrian/vehicle, employ emergency braking 
    overtake_counter = 0         #  a counter to add hysteresis to overtake decision; if negative, never employ this mode
    tailgate_counter = 0         #  a counter to add hysteresis to lane change decision given a tailgating vehicle;, if negative, never employ this mode

    TODO_gains_here = 0


class MPCAgent(BehaviorAgent):
	"""
	Doc TODO.
	"""

	def __init__(self, vehicle, behavior, ignore_traffic_light=False):
		super(MPCAgent, self).__init__(vehicle, ignore_traffic_light=ignore_traffic_light, behavior=behavior)

		# We choose to inherit the high level routing and decision-making functions of the BehaviorAgent.
		# Please take a look at the BehaviorAgent docstring for details, essentially the client calls the following methods:
		# __init__, set_destination, update_information, and run_step

		# This class simply adjusts the low-level control portion via overriding run_step.

	def set_destination(self, start_location, goal_location, debug=False):
		super(MPCAgent, self).set_destination(ego_vehicle.get_location(), spawn_points[spawn_ind_dest].location)

		way_s, way_xy, way_yaw = vtu.extract_path_from_waypoints( self._local_planner.waypoints_queue)
		self._frenet_traj = vtu.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5, debug=debug)

	def run_step(self, debug=False):
		# TODO: think about how I want to modify this state machine.  Try to draw out the resulting behavior.
	    control = None
	    if self.behavior.tailgate_counter > 0:
	        self.behavior.tailgate_counter -= 1
	    if self.behavior.overtake_counter > 0:
	        self.behavior.overtake_counter -= 1

	    ego_vehicle_loc = self.vehicle.get_location()
	    ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

	    # 1: Red lights and stops behavior

	    if self.traffic_light_manager(ego_vehicle_wp) != 0:
	        return self.emergency_stop()

	    # 2.1: Pedestrian avoidancd behaviors

	    walker_state, walker, w_distance = self.pedestrian_avoid_manager(
	        ego_vehicle_loc, ego_vehicle_wp)

	    if walker_state:
	        # Distance is computed from the center of the two cars,
	        # we use bounding boxes to calculate the actual distance
	        distance = w_distance - max(
	            walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
	                self.vehicle.bounding_box.extent.y, self.vehicle.bounding_box.extent.x)

	        # Emergency brake if the car is very close.
	        if distance < self.behavior.braking_distance:
	            return self.emergency_stop()

	    # 2.2: Car following behaviors
	    vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(
	        ego_vehicle_loc, ego_vehicle_wp)

	    if vehicle_state:
	        # Distance is computed from the center of the two cars,
	        # we use bounding boxes to calculate the actual distance
	        distance = distance - max(
	            vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
	                self.vehicle.bounding_box.extent.y, self.vehicle.bounding_box.extent.x)

	        # Emergency brake if the car is very close.
	        if distance < self.behavior.braking_distance:
	            return self.emergency_stop()
	        else:
	            control = self.car_following_manager(vehicle, distance)

	    # 4: Intersection behavior

	    # Checking if there's a junction nearby to slow down
	    elif self.incoming_waypoint.is_junction and (self.incoming_direction == RoadOption.LEFT or self.incoming_direction == RoadOption.RIGHT):
	        control = self._local_planner.run_step(
	            target_speed=min(self.behavior.max_speed, self.speed_limit - 5), debug=debug)

	    # 5: Normal behavior

	    # Calculate controller based on no turn, traffic light or vehicle in front
	    else:
	        control = self._local_planner.run_step(
	            target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)

	    return control


if __name__ == '__main__':
	import carla
	import random
	import numpy as np
	import matplotlib.pyplot as plt

	actor_list = []
	try:
		client = carla.Client('localhost', 2000)
		client.set_timeout(2.0)
		world = client.get_world()
		spawn_points = world.get_map().get_spawn_points()
		
		blueprint_library = world.get_blueprint_library()
		bp = random.choice(blueprint_library.filter('vehicle'))
		
		spawn_inds = np.arange(len(spawn_points))
		spawn_ind_ego = random.choice(spawn_inds)
		ego_vehicle = world.spawn_actor(bp, spawn_points[spawn_ind_ego])
		actor_list.append(ego_vehicle)

		behavior = MPCBehavior()
		agent = MPCAgent(ego_vehicle, behavior)

		spawn_ind_dest = random.choice( np.delete(spawn_inds, spawn_ind_ego) )

		agent.set_destination(ego_vehicle.get_location(), spawn_points[spawn_ind_dest].location, debug=True)
		agent.run_step()
		
		
		

	finally:
		print('destroying actors')
		client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
		print('done.')
