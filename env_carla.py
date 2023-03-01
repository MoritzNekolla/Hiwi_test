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


FIXED_DELTA_SECONDS = 0.1
SUBSTEP_DELTA = 0.01
MAX_SUBSTEPS = 10
EPISODE_TIME = 30


class Environment:
    def __init__(self, world="Town02", host="tks-harper.fzi.de", port=2000):
        self.client = carla.Client(host, port)  # Connect to server
        self.client.set_timeout(10.0)
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

        w_settings = self.world.get_settings()
        w_settings.synchronous_mode = True
        w_settings.fixed_delta_seconds = (
            FIXED_DELTA_SECONDS  # 10 fps | fixed_delta_seconds <= max_substep_delta_time * max_substeps
        )
        w_settings.substepping = True
        w_settings.max_substep_delta_time = SUBSTEP_DELTA
        w_settings.max_substeps = MAX_SUBSTEPS
        self.world.apply_settings(w_settings)
        self.fps_counter = 0
        self.max_fps = int(1 / FIXED_DELTA_SECONDS) * EPISODE_TIME

        print(
            f"~~~~~~~~~~~~~~\n## Simulator settings ##\nFrames: {int(1/FIXED_DELTA_SECONDS)}\nSubstep_delta: {SUBSTEP_DELTA}\nMax_substeps: {MAX_SUBSTEPS}\n~~~~~~~~~~~~~~"
        )

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

        self.tick_world(times=10)
        self.fps_counter = 0

        # time.sleep(
        #     RESET_SLEEP_TIME
        # )  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

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

        self.tick_world()

        # Get velocity of vehicle
        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Set reward and 'done' flag
        done = False
        if len(self.collision_hist) != 0:
            print("Collided")
            done = True
            reward = -200
        elif v_kmh < 20:
            reward = -1
        else:
            reward = 1

        return self.get_observation(), reward, done, None

    def getFPS_Counter(self):
        return self.fps_counter

    def isTimeExpired(self):
        if self.fps_counter > self.max_fps:
            return True
        return False

    # perform a/multiple world tick
    def tick_world(self, times=1):
        for x in range(times):
            self.world.tick()
            self.fps_counter += 1

    def get_observation(self):
        """Observations in PyTorch format BCHW"""
        image = self.observation.transpose((2, 0, 1))  # from HWC to CHW
        image = np.ascontiguousarray(image, dtype=np.float32) / 255
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)  # BCHW
        return image

    def close(self):
        print("Close")

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
