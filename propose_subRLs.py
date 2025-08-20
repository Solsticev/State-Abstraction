from utils import llm_prompt
from utils import llm_utils
import os
import ast


wrapper_template = """
class {}Wrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["{}"]
            if num_item > self.prev_count:
                reward += 1000
                # done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "{}":
                reward += 1000

        return obs, reward, done, info

"""


def define_training_wrapper(obj_name):
    
    return wrapper_template.format(obj_name, obj_name, obj_name)



if __name__ == "__main__":

    config = {
        "rules_path": os.path.join("temp_result", "rules.txt"),
        "plan_path": os.path.join("temp_result", "plan.txt"),
        "goal_list_path": os.path.join("temp_result", "goal_list.txt"),
        "save_model_info": True, 
        "save_model_info_path": os.path.join("temp_result", "submodels1.json"),
        "goal": "collect an iron",
        "save_wrappers": True,
        "save_wrappers_path": os.path.join("temp_result", "submodel_wrappers1.py"),
            }

    rules = open(config["rules_path"], 'r').read()
    plan = open(config["plan_path"], 'r').read()

    goal_list_path = config["goal_list_path"] 
    with open(goal_list_path) as f:
        goal_list_string = f.read()
    goal_list = ast.literal_eval(goal_list_string) 
    

    model_info_dict = llm_utils.llm_chat(prompt = llm_prompt.compose_submodel_prompt(rules, plan, config["goal"]), system_prompt="", model="deepseek-chat")

    if config["save_model_info"]:
        with open(config["save_model_info_path"], 'w') as f:
            f.write(model_info_dict)

    print("model information successfully saved at ", config["save_model_info_path"])
    print("models to train: ")
    print(model_info_dict)

    lines = '''import gym
import numpy as np

def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""

    '''

    if config["save_wrappers"]:

        with open(config["save_wrappers_path"], 'w') as f:
            f.write(lines)
        print("reward shaping wrappers successful saved at ", config["save_wrappers_path"])

        for obj in goal_list:
            with open(config["save_wrappers_path"], 'a') as f:
                f.write(define_training_wrapper(obj))
