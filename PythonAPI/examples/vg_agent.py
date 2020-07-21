import sys
sys.path.append("")
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
import vg_trajectory_utils as vtu

from automatic_control import HUD, World, KeyboardControl
from ltv_kinematic_mpc import LTVKinematicMPC

import pdb

import argparse
import pygame
import logging
import random
import carla
import numpy as np
import time
import matplotlib.pyplot as plt

class MPCBehavior(object):
    """ This is based on the Normal Agent. """

    max_speed = 50               # (km/h); max speed this agent will drive 
    speed_lim_dist = 3           # (km/h); how much slower than the speed limit to drive, if speed limit constraint is active 
    speed_decrease = 10          # (km/h); speed drop when under the TTC threshold 
    safety_time = 3              # (sec); TTC threshold, if we are under this we should slow down 
    min_proximity_threshold = 10 # (m); minimum distance a vehicle should be to consider a possible hazard, i.e. follow it or change lanes 
    braking_distance = 5         # (m); if we are this close to a pedestrian/vehicle, employ emergency braking 
    overtake_counter = 0         #  a counter to add hysteresis to overtake decision; if negative, never employ this mode
    tailgate_counter = 0         #  a counter to add hysteresis to lane change decision given a tailgating vehicle;, if negative, never employ this mode

    TODO_gains_here = 0


class MPCAgent(BehaviorAgent):
    """
    Doc TODO.
    The following are called from automatic_control.py:
    constructor, set_destination, reroute,  update_information, run_step, and need to modify local_planner waypoint queue
    """

    def __init__(self, vehicle, behavior, ignore_traffic_light=False):
        super(MPCAgent, self).__init__(vehicle, ignore_traffic_light=ignore_traffic_light, behavior=behavior)
        self.mpc = LTVKinematicMPC()
        self.max_steer_angle = np.radians( self.vehicle.get_physics_control().wheels[0].max_steer_angle )
        
        self.mpc_ref      = np.zeros((self.mpc.horizon, self.mpc.nx))
        self.prev_mpc_sol = np.zeros((self.mpc.horizon, self.mpc.nu))

        # We choose to inherit the high level routing and decision-making functions of the BehaviorAgent.
        # Please take a look at the BehaviorAgent docstring for details, essentially the client calls the following methods:
        # __init__, set_destination, update_information, and run_step

        # This class simply adjusts the low-level control portion via overriding run_step.

    def set_destination(self, start_location, goal_location, clean=False, debug=False):
        super(MPCAgent, self).set_destination(start_location, goal_location, clean=clean)

        way_s, way_xy, way_yaw = vtu.extract_path_from_waypoints( self._local_planner.waypoints_queue)
        self._frenet_traj = vtu.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.1, debug=debug, viz=True)

        self.control_prev = carla.VehicleControl()
        self.ego_s = -1.0 # setting it negative until it gets feedback in run step so we initially move

    def reroute(self, spawn_points, debug=False):
        super(MPCAgent, self).reroute(spawn_points) # TODO: check the start point!

        way_s, way_xy, way_yaw = vtu.extract_path_from_waypoints( self._local_planner.waypoints_queue )
        self._frenet_traj.update(way_s, way_xy, way_yaw, s_resolution=0.5, viz=True)

    def update_information(self, world):
        super(MPCAgent, self).update_information(world)

    def reached_destination(self, eps=5.0):
        if self.ego_s >= self._frenet_traj.trajectory[-1,0] - eps:
            return True

    def run_step(self, debug=False):
        ego_vehicle_loc   = self.vehicle.get_location()
        ego_vehicle_wp    = self._map.get_waypoint(ego_vehicle_loc)
        ego_vehicle_tf    = self.vehicle.get_transform()
        ego_vehicle_accel = self.vehicle.get_acceleration()

        self.ego_x, self.ego_y = ego_vehicle_loc.x, -ego_vehicle_loc.y
        self.ego_psi = -vtu.fix_angle(np.radians(ego_vehicle_tf.rotation.yaw))

        self.ego_s, self.ego_ey, self.ego_epsi = \
            self._frenet_traj.convert_global_to_frenet_frame(self.ego_x, self.ego_y, self.ego_psi)
        self.ego_curv = self._frenet_traj.get_curvatures_at_s(
                             np.array([self.ego_s + k * max(self.speed / 3.6, 5.0) * self.mpc.dt for k in range(self.mpc.horizon)])
                             )

        self.accel = np.cos(self.ego_psi) * ego_vehicle_accel.x - np.sin(self.ego_psi)*ego_vehicle_accel.y

        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False    

        # Step 1: Generate reference by identifying a max speed based on curvature + stoplights.
        lat_accel_max = 2.0
        max_curv_magnitude = np.max(np.abs(self.ego_curv[:2]))
        if max_curv_magnitude > 0.01:
            max_speed = 3.6 * np.sqrt(lat_accel_max / max_curv_magnitude)            
            max_speed = min(max_speed, self.speed_limit)
        else:
            max_speed = self.speed_limit
        
        if self.traffic_light_manager(ego_vehicle_wp):
            max_speed = 0.0

        #print('max_speed: %.2f, ego_s: %.2f, ego_ey: %.2f, ego_epsi: %.2f' % (max_speed, self.ego_s, self.ego_ey, self.ego_epsi))
        #print('\tego_curv: %.3f, ego_curv_max: %.3f' % (self.ego_curv[0], max_curv_magnitude))

        self.mpc_ref[:, 3] = max_speed / 3.6

        # Step 2: Find linearization trajectory based on current input profile.
        init_state    = np.array([self.ego_s, self.ego_ey, self.ego_epsi, self.speed])
        inputs_lin    = self.prev_mpc_sol
        states_lin    = self.mpc.simulate(init_state, inputs_lin, self.ego_curv, return_with_init_state = True)[:-1]
        Als, Bls, gls = self.mpc.linearize(states_lin, inputs_lin, self.ego_curv, debug=False)
        
        # Step 3: Update the problem parameters and solve.
        self.mpc.update(self.mpc_ref, init_state, self.prev_mpc_sol[0,:], states_lin, inputs_lin, Als, Bls, gls, self.ego_curv)
        states_sol_mpc, inputs_sol_mpc = self.mpc.solve()

        u_acc, u_beta = self.prev_mpc_sol[0,:]
        control.steer = - 2.0 * u_beta / self.max_steer_angle

        k_v = 0.1
        if max_speed == 0.0:
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 1.0
        elif self.speed > max_speed + 2.0:
            control.throttle = 0.0
            control.brake = k_v * (self.speed - self.speed_limit)
        elif self.speed < max_speed - 2.0:
            control.throttle = k_v * (self.speed_limit - self.speed)
            control.brake    = 0.0
        else:
            control.throttle = 0.1
            control.brake    = 0.0

        self.prev_mpc_sol = inputs_sol_mpc

        # Previous PID/FBFF approach:
        # if self.traffic_light_manager(ego_vehicle_wp):
        #     control.throttle = 0.0
        #     control.brake    = 0.5
            
        #     print('light state: ', self.light_state)

        # else:
        #     k_v = 0.1
        #     if self.speed > max_speed + 2.0:
        #         control.throttle = 0.0
        #         control.brake = k_v * (self.speed - self.speed_limit)
        #     elif self.speed < max_speed - 2.0:
        #         control.throttle = k_v * (self.speed_limit - self.speed)
        #         control.brake    = 0.0
        #     else:
        #         control.throttle = 0.1
        #         control.brake    = 0.0
        # k_ey = 0.2
        # x_la = 5.0
        # if self.speed > 30:
        #     k_ey = 0.05
        #     x_la = 15.0
        # control.steer = k_ey * (self.ego_ey + x_la * self.ego_epsi) / self.max_steer_angle

        alpha = 0.8
        
        if control.throttle > 0.0:
            control.throttle = alpha * control.throttle + (1. - alpha) * self.control_prev.throttle
        
        elif control.brake > 0.0:
            control.brake    = alpha * control.brake    + (1. - alpha) * self.control_prev.brake

        control.steer    = alpha * control.steer    + (1. - alpha) * self.control_prev.steer

        control.throttle = np.clip(control.throttle, 0.0, 1.0)
        control.brake    = np.clip(control.brake, 0.0, 1.0)
        control.steer    = np.clip(control.steer, -1.0, 1.0)

        self.control_prev = control        
        return control

    def traffic_light_manager(self, waypoint):
        return super(MPCAgent, self).traffic_light_manager(waypoint)

    '''
    def pedestrian_avoid_manager(self, waypoint):
        walker_list = self._world.get_actors().filter("*walker.pedestrian*")

        interacting_walkers = []

        for w in walker_list:
            wloc = w.get_location() 
            w_x, w_y = wloc.x, wloc.y

            if np.linalg.norm([self.ego_x - w_x, self.ego_y - w_y] < todo_prox_threshold):
                w_s, w_ey, _ = self._frenet_traj.convert_global_to_frenet_frame(w_x, w_y, 0.0)

                if w_s >= self.ego_s and np.abs(w_s - self.ego_s < todo_second_prox_threshold):
                    extent = max(w.bounding_box.extent.y, w.bounding_box.extent.x)
                    interacting_walkers.append([w_s, w_ey, extent])
        del walker_list

        if len(interacting_walkers) == 0:
            return False, None
        else:
            interacting_walkers = np.array(interacting_walkers)
            closest_walker_in_front = np.argmin(interacting_walkers[:,0])
            return True, interacting_walkers[closest_walker_in_front, :]

    def vehicle_avoid_manager(self, waypoint):
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        interacting_vehicles = []

        for vehicle in vehicle_list:
            vloc = vehicle.get_location()
            v_x, v_y = vloc.x, vloc.y

            if np.linalg.norm([self.ego_x - v_x, self.ego_y - v_y] < todo_prox_threshold):
                v_s, v_ey, _ = self._frenet_traj.convert_global_to_frenet_frame(v_x, v_y, 0.0)
    '''

def game_loop(args):
    """ Main loop for agent"""

    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        agent = MPCAgent(world.player, behavior=MPCBehavior())

        spawn_points = world.map.get_spawn_points()
        random.shuffle(spawn_points)

        if spawn_points[0].location != agent.vehicle.get_location():
            destination = spawn_points[0].location
        else:
            destination = spawn_points[1].location

        agent.set_destination(agent.vehicle.get_location(), destination, clean=True, debug=True)

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events():
                return

            # As soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue
    
            agent.update_information(world)

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # Set new destination when target has been reached
            if agent.reached_destination() and args.loop:
                agent.reroute(spawn_points)
                tot_target_reached += 1
                world.hud.notification("The target has been reached " +
                                       str(tot_target_reached) + " times.", seconds=4.0)

            elif agent.reached_destination() and not args.loop:
                print("Target reached, mission accomplished...")
                break

            speed_limit = world.player.get_speed_limit()
            agent.get_local_planner().set_speed(speed_limit)

            control = agent.run_step()
            world.player.apply_control(control)

    finally:
        if world is not None:
            world.destroy()

        pygame.quit()

if __name__ == '__main__':
    """Main method"""
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client with MPCAgent')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.lincoln*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    # argparser.add_argument(
    #     '-b', '--behavior', type=str,
    #     choices=["cautious", "normal", "aggressive"],
    #     help='Choose one of the possible agent behaviors (default: normal) ',
    #     default='normal')
    # argparser.add_argument("-a", "--agent", type=str,
    #                        choices=["Behavior", "Roaming", "Basic"],
    #                        help="select which agent to run",
    #                        default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')