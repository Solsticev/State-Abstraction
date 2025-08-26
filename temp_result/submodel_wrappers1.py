import gym
import numpy as np

def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""

    
class woodWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=-1, decay_steps=500000):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        self.current_step += 1

        obs, reward, done, info = self.env.step(action)

        if self.current_step < self.decay_steps:
            decay_factor = 1.0 - (self.current_step / self.decay_steps)
        else:
            decay_factor = 0.0

        reward *= decay_factor

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        if not self.find_target:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_target = True
                        break

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["wood"]
            if num_item > self.prev_count:
                reward += 10000
                done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "wood":
                reward += 10000
                done = True

        return obs, reward, done, info


class wood_pickaxeWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=-1, decay_steps=500000):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        self.current_step += 1

        obs, reward, done, info = self.env.step(action)

        if self.current_step < self.decay_steps:
            decay_factor = 1.0 - (self.current_step / self.decay_steps)
        else:
            decay_factor = 0.0

        reward *= decay_factor

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        if not self.find_target:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_target = True
                        break

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["wood_pickaxe"]
            if num_item > self.prev_count:
                reward += 10000
                done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "wood_pickaxe":
                reward += 10000
                done = True

        return obs, reward, done, info


class stoneWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=3, decay_steps=500000):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        self.current_step += 1

        obs, reward, done, info = self.env.step(action)

        if self.current_step < self.decay_steps:
            decay_factor = 1.0 - (self.current_step / self.decay_steps)
        else:
            decay_factor = 0.0

        reward *= decay_factor

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        if not self.find_target:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_target = True
                        break

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["stone"]
            if num_item > self.prev_count:
                reward += 10000
                done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "stone":
                reward += 10000
                done = True

        return obs, reward, done, info


class stone_pickaxeWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=-1, decay_steps=500000):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        self.current_step += 1

        obs, reward, done, info = self.env.step(action)

        if self.current_step < self.decay_steps:
            decay_factor = 1.0 - (self.current_step / self.decay_steps)
        else:
            decay_factor = 0.0

        reward *= decay_factor

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        if not self.find_target:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_target = True
                        break

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["stone_pickaxe"]
            if num_item > self.prev_count:
                reward += 10000
                done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "stone_pickaxe":
                reward += 10000
                done = True

        return obs, reward, done, info


class ironWrapper(gym.Wrapper):

    def __init__(self, env, target_obj=9, decay_steps=500000):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        self.current_step += 1

        obs, reward, done, info = self.env.step(action)

        if self.current_step < self.decay_steps:
            decay_factor = 1.0 - (self.current_step / self.decay_steps)
        else:
            decay_factor = 0.0

        reward *= decay_factor

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        if not self.find_target:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_target = True
                        break

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["iron"]
            if num_item > self.prev_count:
                reward += 10000
                done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "iron":
                reward += 10000
                done = True

        return obs, reward, done, info

