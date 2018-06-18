# -*- coding:utf-8 -*-
#!/usr/bin/python

import numpy as np
import os
import cv2
import json

import gym
from gym import error, spaces
from gym.utils import seeding
from constants import DATA_DIR, ID_TO_NAME, TIME_PENALTY_COEF, \
    VISIBILITY_REWARD_SCALE, HALT_STEP, HALT_VISIBILITY

class RobotArmEnv(gym.Env):
    def __init__(self, mode='discrete'):
        self.mode = mode
        [self.base, self.offset, self.latitude, self.longitude] = self.seed()
        if mode == 'discrete':
            self._action_set = [[10,1], [10,-1], [-10,-1], [-10,1]]
            self.action_space = spaces.Discrete(len(self._action_set))
        elif mode == 'continuous':
            self._action_set = [[10],[1]]
            self.action_space = spaces.Discrete(len(self._action_set))

        # RGB (480, 640, 3)  Depth (480, 640, 1)
        (screen_width,screen_height) = (480, 640)

        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        self.num_steps = 0
        self.done = False
        self.visibility = 0
        self.reward = 0.0
        self.viewer = None

    def seed(self):
        # TODO: range of `base` remain to be changed if the dataset is changed
        base = np.random.randint(low=0, high=1)
        offset = np.random.randint(low=0, high=8)

        latitude = np.random.choice(a=[30,40,50,60,70])
        longitude = np.random.choice(a=36)

        return [base, offset, latitude, longitude]

    def step(self, a):
        self.num_steps += 1
        previous_visibility = self.visibility
        if self.mode == 'discrete':
            action = self._action_set[a]

            self.latitude = np.clip(action[0] + self.latitude, 30, 70)
            self.longitude = np.clip(action[1] + self.longitude, 0, 35)
        elif self.mode == 'continuous':
            self.latitude = np.clip(self._round_action_lat(a[0]) + self.latitude, 30, 70)
            self.longitude = np.clip(self._round_action_lon(a[1]) + self.longitude, 0, 35)

        with open(DATA_DIR+'/visibility.json', 'r') as f:
            self.visibility = json.load(f)[ID_TO_NAME[self.base]]
        self.reward = self._get_reward(previous_visibility)
        ob = self._get_obs()
        if self.num_steps >= HALT_STEP or self.visibility >= HALT_VISIBILITY:
            self.done = True
        info = "TODO"

        return ob, self.reward, self.done, info

    def _get_rgb_image(self):
        rgb_image = cv2.imread(DATA_DIR+"/{0:04d}/RGB/{1}_RGB_{2}.jpg"
                               .format(self.base*8+self.offset, self.latitude, self.longitude))
        return rgb_image

    def _get_depth_image(self):
        depth_image = np.load(DATA_DIR+"/{0:04d}/depth/{1}_depth_{2}.npy"
                               .format(self.base*8+self.offset, self.latitude, self.longitude))
        return depth_image

    def _get_target_image(self):
        target_image = cv2.imread(DATA_DIR+"/{0:04d}/RGB/{1}_RGB_{2}.jpg"
                               .format(self.base*8, self.latitude, self.longitude))
        target_image = target_image[target_image.shape[0]//2-100:target_image.shape[0]//2+100,
                       target_image.shape[1]//2-100:target_image.shape[1]//2+100,
                       :]
        return target_image

    def _get_reward(self, previous_visibility):
        visibility_reward = (self.visibility - previous_visibility) * VISIBILITY_REWARD_SCALE
        final_reward = visibility_reward - TIME_PENALTY_COEF
        return final_reward

    def _round_action_lon(self, action_value):
        action_value = np.floor(action_value * 35)
        latitude_shift = round(action_value)
        return latitude_shift

    def _round_action_lat(self, action_value):
        action_value = np.floor(action_value * 20)
        action_value = round(action_value / 10) * 10
        return action_value

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        img_tuple = [self._get_rgb_image(),
                     self._get_depth_image(),
                     self._get_target_image()]
        return img_tuple

    def reset(self):
        self.num_steps = 0
        self.done = False
        self.visibility = 0
        self.reward = 0.0
        [self.base, self.offset, self.latitude, self.longitude] = self.seed()
        return self._get_obs()

    def render(self, mode='human'):
        img = self._get_rgb_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)

ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}

if __name__ == '__main__' :
    env = RobotArmEnv()
    env.reset()
    '''
    # Level 1: Getting environment up and running
    for _ in range(1000): # run for 1000 steps
        # env.render()
        action = env.action_space.sample() # pick a random action
        env.step(action) # take action
    '''

    # Level 2: Running trials(AKA episodes)
    for i_episode in range(20):
        observation = env.reset()  # reset for each new trial
        for t in range(100):  # run for 100 timesteps or until done, whichever is first
            env.render()
            action = env.action_space.sample()  # select a random action (see https://github.com/openai/gym/wiki/CartPole-v0)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    '''
    # Level 3: Non-random actions
    highscore = 0
    for i_episode in range(20):  # run 20 episodes
        observation = env.reset()
        points = 0  # keep track of the reward each episode
        while True:  # run until episode is done
            env.render()
            action = 1 if observation[2] > 0 else 0  # if angle if positive, move right. if angle is negative, move left
            observation, reward, done, info = env.step(action)
            points += reward
            if done:
                if points > highscore:  # record high score
                    highscore = points
                    break
    '''
    



