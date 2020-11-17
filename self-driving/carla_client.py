import random
import weakref

import cv2
import pygame
import numpy as np

import carla

from utils import pre_processing

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

class CarlaClient():

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None
        self.image = None
        self.display = None
        self.capture = True

    def setup(self):
        try:
            pygame.init()
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        except Exception as e:
            print(e)

    def setdown(self):
        self.set_synchronous_mode(False)
        self.camera.destory()
        self.car.destory()
        pygame.quit()
        cv2.destroyAllWindows()

        
    def camera_bp(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))

        return camera_bp

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
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

    @staticmethod
    def set_image(weak_self, img):
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            # remove info do not need
            array = pre_processing(array)

            #array = array[:, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    
if __name__ == "__main__":
    cc = CarlaClient()
    cc.setup()
    cc.setup_car()
    cc.setup_camera()
    pygame_clock = pygame.time.Clock()
    cc.car.set_autopilot(True)

    while True:
        cc.world.tick()
        cc.capture = True
        pygame_clock.tick_busy_loop(30)
        cc.render(cc.display)
        pygame.display.flip()
        pygame.event.pump()
        cv2.waitKey(1)


