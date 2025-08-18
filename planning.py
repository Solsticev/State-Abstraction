import llm_prompt
import llm_utils
import ast
import os

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
        if np.array(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["{}"]
            if num_item > self.prev_count:
                reward += 1000
                # done = True
        except KeyError as e:
            if face_at(info["obs"]) == "{}":
                reward += 1000
        self.prev_count = num_item

        return obs, reward, done, info

"""


def define_training_wrapper(obj_name):
    
    return wrapper_template.format(obj_name, obj_name, obj_name)


def plan(tasks_list, num_step, rules):

    return plan_aux(tasks_list, [], 0, num_step, rules)


def plan_aux(tasks_list, command_list, current_step, num_step, rules):

    if current_step == num_step or len(tasks_list) == 0:
        return tasks_list, command_list

    current_tasks_list = tasks_list

    tasks_list = []
    command_list = []

    for subgoal in current_tasks_list:

        PLANNING_PROMPT = llm_prompt.compose_planning_prompt(rules)

        response = llm_utils.llm_chat(prompt="Current goal: " + subgoal, system_prompt=PLANNING_PROMPT, model="deepseek-chat")
        subgoals_list = []
        try: 
            llm_subgoals_list = ast.literal_eval(response)
            for new_subgoal in llm_subgoals_list:
                response = llm_utils.llm_chat(prompt=new_subgoal,system_prompt=llm_prompt.TRANS_PROMPT, model="deepseek-chat")
                if "None" not in response and response not in command_list:
                    tasks_list.append(new_subgoal)
                    command_list.append(response)

        except Exception as e:
            pass

    print(tasks_list)
    print(command_list)

    return plan_aux(tasks_list, command_list, current_step+1, num_step, rules)


if __name__ == "__main__":

    config = {"rules_path": "rules.txt",
              "tasks_list": ["collect an iron"],
              "planning_steps": 3,
              "save_wrappers": False,
              "save_plan": False,
              "save_wrappers_path": "submodel_wrappers.py",
              "save_plan_path": "plan.txt"
              }

    rules = open(config["rules_path"], 'r').read()
    tasks_list = config["tasks_list"]
    planning_steps = config["planning_steps"]
    
    tasks_list, command_list = plan(tasks_list, planning_steps, rules)

    print(tasks_list)
    print(command_list)


    lines = '''import gym
import numpy as np

def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""

    '''

    if config["save_plan"]:

        with open(config["save_plan_path"], 'w') as f:
            f.write(str(tasks_list))
    
    if config["save_wrappers"]:

        with open(config["save_wrappers_path"], 'w') as f:
            f.write(lines)

        print("LLM generated plans: ")
        print(command_list)

        for obj in command_list:
            with open(config["save_wrappers_path"], 'a') as f:
                f.write(define_training_wrapper(obj))
