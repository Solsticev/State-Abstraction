from utils import llm_prompt
from utils import llm_utils
import os
import ast
import json
import argparse


OBJ_INDEX = {"water": 1, "grass": 2, "stone": 3, "path": 4, "sand": 5, "tree": 6, "lava": 7, "coal": 8, "iron": 9, "diamond": 10, "table": 11, "furnace": 12}


wrapper_template = """
class {}Wrapper(gym.Wrapper):

    def __init__(self, env, target_obj={}, decay_steps=500000):
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
            num_item = info["inventory"]["{}"]
            if num_item > self.prev_count:
                reward += 10000
                done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "{}":
                reward += 10000
                done = True

        return obs, reward, done, info

"""


def define_training_wrapper(obj_name):

    if obj_name in OBJ_INDEX:
        return wrapper_template.format(obj_name, OBJ_INDEX[obj_name], obj_name, obj_name)
    else:
        return wrapper_template.format(obj_name, -1, obj_name, obj_name)


def check_valid(model_info_dict_str):

    valid_items = {"health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword", "furnace", "table"}

    try:
        model_info_dict = json.loads(model_info_dict_str)
        if len(model_info_dict) == 0:
            print("no sub model returned! retrying...")
            return False
        if "model0" in model_info_dict.keys():
            print("model name should start with model1, not model0! Retrying....")
            return False
        for value in model_info_dict.values():
            if "name" not in value.keys() or "description" not in value.keys() or "requirement" not in value.keys():
                print("information of sub model is incomplete! Retrying...")
                return False
            if value["name"] not in valid_items:
                print("invalid wrapper name! Retrying...")
                return False
            reqs = value["requirement"]
            for req in reqs:
                if req[0] not in valid_items or len(req) != 2:
                    return False

    except Exception as e:
        return False
    
    return True


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--rules_path", type=str, default=os.path.join("temp_result", "rules.txt"))
    parser.add_argument("--plan_path", type=str, default=os.path.join("temp_result", "plan.txt"))
    parser.add_argument("--save_model_info_path", type=str)
    parser.add_argument("--final_task", type=str)
    parser.add_argument("--save_wrappers_path", type=str)

    args = parser.parse_args()

    config = {
        "rules_path": args.rules_path,
        "plan_path": args.plan_path,
        "save_model_info": True, 
        "save_model_info_path": args.save_model_info_path,
        "goal": args.final_task,
        "save_wrappers": True,
        "save_wrappers_path": args.save_wrappers_path,
            }

    rules = open(config["rules_path"], 'r').read()
    plan = open(config["plan_path"], 'r').read()
    
    max_retries = 10
    for i in range(max_retries):

        model_info_dict = llm_utils.llm_chat(prompt = llm_prompt.compose_submodel_prompt(rules, plan, config["goal"]), system_prompt="", model="deepseek-chat")

        is_valid = check_valid(model_info_dict)
        if not is_valid and i == max_retries-1:
            print("LLM output is invalid!")
            assert False
        elif is_valid:
            break

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

        model_info = json.loads(model_info_dict)

        for obj_info in model_info.values():
            obj = obj_info["name"] 
            with open(config["save_wrappers_path"], 'a') as f:
                f.write(define_training_wrapper(obj))
