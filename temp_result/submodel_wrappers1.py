import gym
import numpy as np
from gym import spaces


def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""

    
class woodWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=-1, allowed_actions=[0, 1, 2, 3, 4, 5]):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.allowed_actions = allowed_actions
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        
        if action not in self.allowed_actions:
            reward -= 10

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.001

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        for i in range(left_index, right_index, 1):
            if not self.find_target:
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 5
                        self.find_target = True
                        break

        self.prev_pos = player_pos

        num_item = info["inventory"]["wood"]
        if num_item > self.prev_count:
            reward += 100
            done = True
        self.prev_count = num_item

        return obs, reward, done, info


class wood_pickaxeWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=-1, allowed_actions=[0, 1, 2, 3, 4, 5, 8, 11]):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.allowed_actions = allowed_actions
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        
        if action not in self.allowed_actions:
            reward -= 10

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.001

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        for i in range(left_index, right_index, 1):
            if not self.find_target:
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 5
                        self.find_target = True
                        break

        self.prev_pos = player_pos

        num_item = info["inventory"]["wood_pickaxe"]
        if num_item > self.prev_count:
            reward += 100
            done = True
        self.prev_count = num_item

        return obs, reward, done, info


class stoneWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=3, allowed_actions=[0, 1, 2, 3, 4, 5, 11, 8]):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.allowed_actions = allowed_actions
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        
        if action not in self.allowed_actions:
            reward -= 10

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.001

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        for i in range(left_index, right_index, 1):
            if not self.find_target:
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 5
                        self.find_target = True
                        break

        self.prev_pos = player_pos

        num_item = info["inventory"]["stone"]
        if num_item > self.prev_count:
            reward += 100
            done = True
        self.prev_count = num_item

        return obs, reward, done, info


class stone_pickaxeWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=-1, allowed_actions=[0, 1, 2, 3, 4, 5, 8, 11, 12]):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.allowed_actions = allowed_actions
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        
        if action not in self.allowed_actions:
            reward -= 10

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.001

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        for i in range(left_index, right_index, 1):
            if not self.find_target:
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 5
                        self.find_target = True
                        break

        self.prev_pos = player_pos

        num_item = info["inventory"]["stone_pickaxe"]
        if num_item > self.prev_count:
            reward += 100
            done = True

        self.prev_count = num_item

        if done:
            print(reward)

        return obs, reward, done, info


class ironWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=9, allowed_actions=[0, 1, 2, 3, 4, 5]):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.allowed_actions = allowed_actions
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        
        if action not in self.allowed_actions:
            reward -= 10

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.001

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        for i in range(left_index, right_index, 1):
            if not self.find_target:
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 5
                        self.find_target = True
                        break

        self.prev_pos = player_pos

        num_item = info["inventory"]["iron"]
        if num_item > self.prev_count:
            reward += 100
            done = True
        self.prev_count = num_item

        return obs, reward, done, info

