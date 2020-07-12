import carla # if it fails, check sys.path for .egg file
import subprocess
import argparse
import time



if __name__ == '__main__':
	num_walkers  = 10
	num_vehicles = 30

	# Configure map/simulation.
	config_str = "python ../util/config.py --map Town05 --weather Default --fps 20"
	ret = subprocess.call(config_str.split(" "))
	print(ret)

	# Spawn vehicles and walkers.
	spawn_str = "python spawn_npc.py -n {0} -w {1}".format( int(num_vehicles), int(num_walkers) )
	spawn_proc = subprocess.Popen(spawn_str.split(" "))
	time.sleep(10.0)

	
	# use automatic_control.py to get access to GUI/World.
	# subclass Agent to make a MPC Agent
	# convert waypoints to Frenet Frame
	# s, v control: use local curvature, obstacles, and speed limit to set s_max and v_max
	# ey, epsi control: regulate to zero using only steering and given s, v schedule
	# simple PID controller to track desired acceleration and steering angle

	

	# Destroy spawned vehicles.
	if spawn_proc:
		spawn_proc.send_signal(subprocess.signal.SIGINT)	

