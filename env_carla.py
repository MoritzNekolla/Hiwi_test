import carla

import random
import numpy as np
import math
import time

import torch

IM_WIDTH = 480
IM_HEIGHT = 360

BEV_DISTANCE = 20

N_ACTIONS = 9

RESET_SLEEP_TIME = 1


class Environment:
    def __init__(self, world="Town02_Opt", host="localhost", port=2000):
        self.client = carla.Client(host, port)  # Connect to server
        self.client.set_timeout(30.0)
        self.world = self.client.load_world(world)
        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        self.actor_list = []
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT

        # Enable synchronous mode between server and client
        # self.settings = self.world.get_settings()
        # self.settings.synchronous_mode = True # Enables synchronous mode
        # self.world.apply_settings(self.settings)

    def init_ego(self):
        self.vehicle_bp = self.bp_lib.find("vehicle.tesla.model3")
        self.ss_camera_bp = self.bp_lib.find("sensor.camera.semantic_segmentation")
        self.col_sensor_bp = self.bp_lib.find("sensor.other.collision")

        # Configure sensors
        self.ss_camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.ss_camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.ss_camera_bp.set_attribute("fov", "110")

        self.ss_cam_location = carla.Location(0, 0, BEV_DISTANCE)
        self.ss_cam_rotation = carla.Rotation(-90, 0, 0)
        self.ss_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)

        self.col_sensor_location = carla.Location(0, 0, 0)
        self.col_sensor_rotation = carla.Rotation(0, 0, 0)
        self.col_sensor_transform = carla.Transform(self.col_sensor_location, self.col_sensor_rotation)

        self.collision_hist = []

    def reset(self):
        for actor in self.actor_list:
            actor.destroy()

        self.actor_list = []
        self.collision_hist = []

        # Spawn vehicle
        transform = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, transform)
        self.actor_list.append(self.vehicle)

        # Attach and listen to image sensor (BEV Semantic Segmentation)
        self.ss_cam = self.world.spawn_actor(
            self.ss_camera_bp, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid
        )
        self.actor_list.append(self.ss_cam)
        self.ss_cam.listen(lambda data: self.__process_sensor_data(data))

        time.sleep(
            RESET_SLEEP_TIME
        )  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        # Attach and listen to collision sensor
        self.col_sensor = self.world.spawn_actor(self.col_sensor_bp, self.col_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.__process_collision_data(event))

        self.episode_start = time.time()
        return self.get_observation()

    def step(self, action):
        # Easy actions: Steer left, center, right (0, 1, 2)
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1))
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-1))
        elif action == 8:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=1))

        # Get velocity of vehicle
        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Set reward and 'done' flag
        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward = -1
        elif v_kmh < 20:
            reward = -1
        else:
            reward = 1

        return self.get_observation(), reward, done, None

    def get_observation(self):
        """Observations in PyTorch format BCHW"""
        image = self.observation.transpose((2, 0, 1))  # from HWC to CHW
        image = np.ascontiguousarray(image, dtype=np.float32) / 255
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)  # BCHW
        return image

    def __process_sensor_data(self, image):
        """Observations directly viewable with OpenCV in CHW format"""
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        self.observation = i3

    def __process_collision_data(self, event):
        self.collision_hist.append(event)

    def __del__(self):
        for actor in self.actor_list:
            actor.destroy()
