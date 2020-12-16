import random
import weakref

import cv2
import pygame
import numpy as np

import sys
sys.path.append("carla-0.9.10-py3.7-linux-x86_64.egg")
import carla

from client_bounding_boxes import ClientSideBoundingBoxes

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

class CarlaClient:

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.image = None
        self.display = None
        self.capture = None
        self.raw_data = None

        # data
        self.image_data = None
        self.lidar_data = None
        self.bboxs = None

    def setup(self):
        try:
            pygame.init()
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
        except Exception as e:
            print(e)

    def setdown(self):
        self.set_synchronous_mode(False)
        self.camera.destory()
        self.car.destory()
        pygame.quit()
        cv2.destoryAllWindows()

    def camera_bp(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        bp.set_attribute('fov', str(VIEW_FOV))

        return bp
    
    def lidar_bp(self):
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '5000')
        bp.set_attribute('upper_fov', '15')
        bp.set_attribute('lower_fov', '-25')
        bp.set_attribute('channels', '64')
        bp.set_attribute('points_per_second', '1000000')

        return bp

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        car_bp = self.world.get_blueprint_library().filter('model3')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera = self.world.spawn_actor(self.camera_bp(), camera_transform, attach_to=self.car)

        # TODO: camera don't has calibration attribute
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration 
        weak_self = weakref.ref(self)
        self.camera.listen(lambda data: weak_self().set_image(weak_self, data))

    def setup_lidar(self):
        lidar_transform = carla.Transform(
                # carla.Location(x=-5.5, z=2.8),
                # carla.Rotation(pitch=-15, yaw=0)
                )
        self.lidar = self.world.spawn_actor(self.lidar_bp(), lidar_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.lidar.listen(lambda data: weak_self().set_lidar(weak_self, data))

    @staticmethod
    def set_image(weak_self, data):
        self = weak_self()
        if self.capture:
            self.image = data # data: carla.CameraMeasurement
            self.capture = False
        self.image_data = data
        self.bboxs = self.get_bb_data()

    @staticmethod
    def set_lidar(weak_self, data):
        self = weak_self()
        self.lidar_data = data # data: carla.LidarMeasurement

    def get_bb_data(self):
        vehicles_on_world = self.world.get_actors().filter('vehicle.*')
        walkers_on_world = self.world.get_actors().filter('walker.*')
        bounding_boxes_vehicles = ClientSideBoundingBoxes.get_bounding_boxes(vehicles_on_world, self.camera)
        bounding_boxes_walkers = ClientSideBoundingBoxes.get_bounding_boxes(walkers_on_world, self.camera)
        return [bounding_boxes_vehicles, bounding_boxes_walkers]


    def render(self, display):
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))


if __name__ == "__main__":
    cc = CarlaClient()
    cc.setup()
    cc.set_synchronous_mode(True)
    cc.setup_car()
    cc.setup_camera()
    cc.setup_lidar()
    pygame_clock = pygame.time.Clock()
    cc.car.set_autopilot(True)

    while True:
        print(cc.lidar_data)
        print(cc.image_data)
        print(cc.bboxs)
        cc.world.tick()
        cc.capture = True
        pygame_clock.tick_busy_loop(30)
        cc.render(cc.display)
        pygame.display.flip()
        pygame.event.pump()
        cv2.waitKey(1)
