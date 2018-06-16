import gym
import json
import numpy as np
from PIL import Image
from gym import error, spaces, utils
from gym.spaces import Tuple, Box, Discrete
from gym.utils import seeding
from main import visibility, max_visibility, lat_list, lon_list


gamma = 0.05
scaling_factor = 0.2
RGB_path = "RobotArm-DA/0003/RGB/"
depth_path = "RobotArm-DA/0003/depth/"
target_path = "RobotArm-DA/0000/RGB/"

target_width = 400
target_height = 300

def round_action_lon(action_value):
    action_value = np.floor(action_value * 35)
    return np.floor(action_value)


def round_action_lat(action_value):
    action_value = np.floor(action_value*20)
    if -5 <= action_value <= 5:
        action_value = 0
    elif 5 < action_value <= 15:
        action_value = 10
    elif -15 <= action_value < -5:
        action_value = -10
    elif 15 < action_value <= 20:
        action_value = 20
    else:
        action_value = -20
    return action_value


class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # self.action_space = Tuple((Discrete(LAT_RANGE), Discrete(LON_RANGE)))  # (latitude, longitude)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        self.observation_space = (spaces.Box(low=0, high=255, shape=(3,)),  # observed RGB
                                  spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # observed depth
                                  spaces.Box(low=0, high=255, shape=(3,)),  # target RGB
                                  )

        self.lat = 10 * np.random.random_integers(3, 7)
        self.lon = np.random.random_integers(0, 35)
        RGB_matrix = np.array(Image.open(str(self.lon) + "_RGB_" + str(self.lat) + ".jpg"))
        depth_matrix = np.load(str(self.lon) + "_depth_" + str(self.lat) + ".npy")
        self.observation = (RGB_matrix, depth_matrix)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        done = False
        self.render()
        im = Image.open(str(self.lon) + "_" + str(self.lat) + ".jpg")
        im.size()
        new_lat = round_action_lat(action[0]) + self.lat
        if new_lat > 70:
            new_lat = 70
        elif new_lat < 30:
            new_lat = 30
        new_lon = (round_action_lon(action[1]) + self.lon) % 36
        if new_lon < 0:
            new_lon += 36
        assert lat_list.contains(new_lat), "invalid latitude!"
        assert lon_list.contains(new_lon), "invalid longitude!"
        RGB_matrix = np.array(Image.open(RGB_path + str(self.lon) + "_RGB_" + str(self.lat) + ".jpg"))  # 480 * 640
        depth_matrix = np.load(depth_path + str(self.lon) + "_depth_" + str(self.lat) + ".npy")

        ima = Image.open(target_path + lon_list[np.random.randint(0,len(lon_list))] + "_RGB_" + lat_list[np.random.randint(0,len(lat_list))] + ".jpg")
        left = np.random.randint(0, 200)
        up = np.random.randint(0, 150)
        box = (left, up, left + target_width, up + target_height)
        target_matrix = np.array(ima.crop(box))  # target_width * target_height

        self.observation = (RGB_matrix, depth_matrix, target_matrix)
        reward = visibility[str(new_lat)][str(new_lon)] - visibility[str(self.lat)][str(self.lon)] - gamma * (action[0] + action[1]*scaling_factor)

        self.lat = new_lat
        self.lon = new_lon
        if  self.observation >= 0.9*max_visibility:
            done = True
        return self.observation, reward, done, {"position": (self.lat,self.lon), "visibility": visibility[str(self.lat)][str(self.lon)]}

    def reset(self):
        self.lat = 10 * np.random.random_integers(3,7)
        self.lon = np.random.random_integers(0,35)
        self.observation = visibility[str(self.lat)][str(self.lon)]

        return self.observation

    def render(self, mode='human', close=False):
        im = Image.open(str(self.lon) + "_RGB_" + str(self.lat) + ".jpg")
        im.show()
