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
EPISODE_TIME = 20


class Environment:
    def __init__(self, world="Town02", host="tks-harper.fzi.de", port=2000, roadGraph=True, spawn_deviation=True, semantic_seg=True):
        self.client = carla.Client(host, port)  # Connect to server
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(world)
        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        self.actor_list = []
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT
        self.agent_transform = None

        self.trajectory_list = None
        self.traj_wp_list = None
        self.roadGraph = roadGraph
        self.goalPoint = None
        self.spawn_deviation = spawn_deviation
        self.semantic_seg = semantic_seg
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
        if self.semantic_seg:
            self.ss_camera_bp = self.bp_lib.find("sensor.camera.semantic_segmentation")
        else:
            self.ss_camera_bp = self.bp_lib.find('sensor.camera.rgb')
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
        spawn_failed = True
        while spawn_failed:
            transform = random.choice(self.spawn_points)
            if self.spawn_deviation:
                prob = random.random()
                if prob < 0.3333:
                    deviation = random.random() * 0.45
                    prob = random.random()
                    if prob < 0.5:
                        transform.location.x += deviation
                        transform.location.y += deviation
                    else:
                        transform.location.x -= deviation
                        transform.location.y -= deviation

                    deviation = random.randrange(0,45)
                    if prob < 0.5:
                        transform.rotation.yaw += deviation
                        if transform.rotation.yaw > 360:
                            transform.rotation.yaw = transform.rotation.yaw - 360
                    else:
                        transform.rotation.yaw -= deviation
                        if transform.rotation.yaw < 0:
                            transform.rotation.yaw = transform.rotation.yaw * (-1)

            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, transform)
            if not self.vehicle == None: spawn_failed = False
        self.actor_list.append(self.vehicle)

        # Attach and listen to image sensor (BEV Semantic Segmentation)
        self.ss_cam = self.world.spawn_actor(self.ss_camera_bp, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.ss_cam)
        self.ss_cam.listen(lambda data: self.__process_sensor_data(data))


        # self.tick_world(times=10) Let's see if we need this anymore in 0.9.14

        # Attach and listen to collision sensor
        self.col_sensor = self.world.spawn_actor(self.col_sensor_bp, self.col_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.__process_collision_data(event))

        
        self.tick_world(times=int(1 / FIXED_DELTA_SECONDS))
        self.fps_counter = 0
        self.agent_transform = self.get_Vehicle_transform()

        self.set_goalPoint()
        if self.roadGraph:
            self.plotTrajectory()
        
        self.episode_start = time.time()

        obs = self.get_observation()

        # print("--------------")
        return obs
    
    def set_goalPoint(self):
        ego_map_point = self.getEgoWaypoint() # closest map point to the spawn point
        # print(ego_map_point)
        tmp_trajectory_list = [ego_map_point]
        self.trajectory_list = [[ego_map_point.transform.location.x, ego_map_point.transform.location.y, ego_map_point.transform.rotation.yaw]]
        self.traj_wp_list = [ego_map_point]
        for x in range(80):
            next_point = tmp_trajectory_list[-1].next(2.)[0]
            # next_point = next_point.next(2.)[0]
            tmp_trajectory_list.append(next_point)
            self.traj_wp_list.append(next_point)
            self.trajectory_list.append([next_point.transform.location.x, next_point.transform.location.y, next_point.transform.rotation.yaw])

        self.goalPoint = self.trajectory_list[-1]

        # for k in self.trajectory_list:
        #     print(k)

        # print("----------")
    # plots the path from spawn to goal
    def plotTrajectory(self):
        stretch = 0.3
        lifetime=0.2
        for x in range(len(self.traj_wp_list)-1):
            w = self.traj_wp_list[x]
            w1 = self.traj_wp_list[x + 1]
            L_w = carla.Location(x=w.transform.location.x,y=w.transform.location.y,z=w.transform.location.z + stretch)
            L_w1 = carla.Location(x=w1.transform.location.x,y=w1.transform.location.y,z=w1.transform.location.z + stretch)
            self.world.debug.draw_line(L_w, L_w1, thickness=0.1, life_time=lifetime, color=carla.Color(r=255, g=255, b=255))
        # color goal point in red 
        L_w = carla.Location(x=self.traj_wp_list[-1].transform.location.x,y=self.traj_wp_list[-1].transform.location.y,z=self.traj_wp_list[-1].transform.location.z + stretch)
        self.world.debug.draw_point(L_w, size=0.3, life_time=lifetime, color=carla.Color(r=255, g=0, b=0))

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
        # if self.roadGraph:
        #     self.plotTrajectory()

        reward, done = self._get_reward()

        # if self.roadGraph and self.fps_counter % 10 == 0:
        #     self.plotTrajectory()

        return self.get_observation(), reward, done, None
    
    def _get_reward(self):
        """Calculate the step reward."""
        done = False

        # reward for speed tracking
        v = self.vehicle.get_velocity()
        # speed = np.sqrt(v.x**2 + v.y**2)
        # r_speed = -abs(speed - self.desired_speed)
        
        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1
            done = True

        # reward for steering:
        r_steer = -self.vehicle.get_control().steer**2

        # reward for out of lane
        ego_pos = self.get_Vehicle_positionVec()
        ego_x = ego_pos[0]
        ego_y = ego_pos[1]
        dis, w = self.get_lane_dis(self.trajectory_list, ego_x, ego_y)
        r_out = 0
        if abs(dis) > 2.0:
            r_out = -1
        
        if abs(dis) > 3.:
            done = True
        
        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)
            # 'out_lane_thres': 2.0,  # threshold for out of lane
            # 'desired_speed': 8,  # desired speed (m/s)
        # cost for too fast
        r_fast = 0
        if lspeed_lon > 8:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.vehicle.get_control().steer) * lspeed_lon**2

        r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1


        dis = d = np.sqrt((ego_x-self.goalPoint[0])**2 + (ego_y-self.goalPoint[1])**2)
        # print(self.trajectory_list[-1])
        # print(ego_pos)
        # print(abs(dis))
        if abs(dis) < 2.0:
            r = 1000
            done = True

        if self.isTimeExpired():
            done = True

        return r, done
    
    def get_lane_dis(self, waypoints, x, y):
        """
        Calculate distance from (x, y) to waypoints.
        :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
        :param x: x position of vehicle
        :param y: y position of vehicle
        :return: a tuple of the distance and the closest waypoint orientation
        """
        dis_min = 999999
        waypt = waypoints[0]
        for pt in waypoints:
            d = np.sqrt((x-pt[0])**2 + (y-pt[1])**2)
            if d < dis_min:
                dis_min = d
                waypt=pt
        vec = np.array([x - waypt[0], y - waypt[1]])
        lv = np.linalg.norm(np.array(vec))
        w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
        cross = np.cross(w, vec/lv)
        dis = - lv * cross
        return dis, w
    

    def getFPS_Counter(self):
        return self.fps_counter

    def isTimeExpired(self):
        if self.fps_counter > self.max_fps:
            return True
        return False

    def getGoalPoint(self):
        return self.goalPoint
    
    def getEgoWaypoint(self):
        # vehicle_loc = self.vehicle.get_location()
        vehicle_loc = self.agent_transform.location
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True,
                      lane_type=carla.LaneType.Driving)

        return wp
    
    # perform a/multiple world tick
    def tick_world(self, times=1):
        for x in range(times):
            self.world.tick()
            self.fps_counter += 1

    #get vehicle location and rotation (0-360 degrees)
    def get_Vehicle_transform(self):
        return self.vehicle.get_transform()
    
    #get vehicle location
    def get_Vehicle_positionVec(self):
        position = self.vehicle.get_transform().location
        return np.array([position.x, position.y, position.z])

    # ## semantic seg
    # def get_observation(self):
    #     """Observations in PyTorch format BCHW"""
    #     self.agent_transform = self.get_Vehicle_transform()
    #     image = self.observation.transpose((2, 0, 1))  # from HWC to CHW
    #     image = np.ascontiguousarray(image, dtype=np.float32) / 255
    #     image = torch.from_numpy(image)
    #     image = image.unsqueeze(0)  # BCHW
    #     return image

    # rgb
    def get_observation(self):
        if self.roadGraph:
            self.plotTrajectory()
        if self.semantic_seg:
            """Observations in PyTorch format BCHW"""
            self.agent_transform = self.get_Vehicle_transform()
            image = self.observation.transpose((2, 0, 1))  # from HWC to CHW
            image = np.ascontiguousarray(image, dtype=np.float32) / 255
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)  # BCHW
            return image
        else: 
            self.agent_transform = self.get_Vehicle_transform()
            frame = self.observation
            frame = frame.astype(np.float32) / 255
            frame = self.arrange_colorchannels(frame)
            frame = np.transpose(frame, (2,0,1))
            frame = torch.as_tensor(frame)
            frame = frame.unsqueeze(0)
            # print(frame.shape)
            return frame

    def close(self):
        print("Close")

    # semantic seg
    # def __process_sensor_data(self, image):
    #     """Observations directly viewable with OpenCV in CHW format"""
    #     image.convert(carla.ColorConverter.CityScapesPalette)
    #     i = np.array(image.raw_data)
    #     i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    #     i3 = i2[:, :, :3]
    #     self.observation = i3

    # rgb
    def __process_sensor_data(self, image):
        if self.semantic_seg:
            """Observations directly viewable with OpenCV in CHW format"""
            image.convert(carla.ColorConverter.CityScapesPalette)
            i = np.array(image.raw_data)
            i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
            i3 = i2[:, :, :3]
            self.observation = i3
        else:
            """ Observations directly viewable with OpenCV in CHW format """
            # image.convert(carla.ColorConverter.CityScapesPalette)
            i = np.array(image.raw_data)
            i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
            i3 = i2[:, :, :3]
            self.observation = i3

    def __process_collision_data(self, event):
        self.collision_hist.append(event)

    # changes order of color channels. Silly but works...
    def arrange_colorchannels(self, image):
        mock = image.transpose(2,1,0)
        tmp = []
        tmp.append(mock[2])
        tmp.append(mock[1])
        tmp.append(mock[0])
        tmp = np.array(tmp)
        tmp = tmp.transpose(2,1,0)
        return tmp

    def __del__(self):
        for actor in self.actor_list:
            actor.destroy()
