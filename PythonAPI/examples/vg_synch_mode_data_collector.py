import carla
import pygame
import random

from synchronous_mode import CarlaSyncMode, draw_image, get_font, should_quit


def main_autopilot(args, camera_config, max_frames=1000):
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter(args.filter)),
            start_pose)
        vehicle.set_autopilot(True)
        ego_id = vehicle.id
        actor_list.append(vehicle)

        sensor_location = carla.Location(x=camera_config['x'], y=camera_config['y'],
                                         z=camera_config['z'])
        sensor_rotation = carla.Rotation(pitch=camera_config['pitch'],
                                         roll=camera_config['roll'],
                                         yaw=camera_config['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        bp_rgb = bp_library.find('sensor.camera.rgb')
        bp_depth = bp_library.find('sensor.camera.depth')
        bp_seg = bp_library.find('sensor.camera.semantic_segmentation')

        for bp in [bp_rgb, bp_depth, bp_seg]:
            bp.set_attribute('image_size_x', str(camera_config['width']))
            bp.set_attribute('image_size_y', str(camera_config['height']))
            bp.set_attribute('fov', str(camera_config['fov']))


        camera_rgb = world.spawn_actor(bp_rgb, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_rgb)
        camera_depth = world.spawn_actor(bp_depth, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_depth)
        camera_seg = world.spawn_actor(bp_seg, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_seg)
        
        # Create a synchronous mode context.
        num_frames_saved = 0
        with CarlaSyncMode(world, camera_rgb, camera_depth, camera_semseg, fps=args.fps) as sync_mode:
            while True:
                if should_quit() or num_frames_saved >= max_frames:
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_depth, image_semseg = sync_mode.tick(timeout=2.0)

                ego_snap = snapshot.find(ego_id)
                vel_ego = ego_snap.get_velocity()
                vel_thresh = 1.0
                if vel_ego.x**2 + vel_ego.y**2 > vel_thresh:
                    # image_rgb.save_to_disk('%s/rgb/%08d' % (args.logdir, num_frames_saved))
                    # image_depth.save_to_disk('%s/depth/%08d' % (args.logdir, num_frames_saved))
                    # image_semseg.save_to_disk('%s/seg/%08d' % (args.logdir, num_frames_saved))
                    num_frames_saved +=1
                    print('Frames Saved: %d of %d' % (num_frames_saved, max_frames))


                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='CARLA Synchronous Camera Data Collector')
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
        default='800x600',
        help='window resolution (default: 800x600)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.lincoln.*',
        help='actor filter (default: "vehicle.lincoln.*")')
    argparser.add_argument( 
        '--logdir',
        default='data_synced',
        help='Image logging directory for saved rgb,depth,and semantic segmentation images.')
    argparser.add_argument( 
        '--fps',
        default=5,
        type=int)
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    camera_config = {'x':0.7, 'y':0.0, 'z':1.60, \
                     'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
                     'width':800, 'height': 600, 'fov':100} 

    try:
        main_autopilot(args, camera_config)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')