from utils import llm_prompt
from utils import llm_utils
import ast
import os


def plan(tasks_list, num_step, rules):

    print("planning...")

    return plan_aux(tasks_list, [], 0, num_step, rules)


def plan_aux(tasks_list, goal_list, current_step, num_step, rules):

    if current_step == num_step or len(tasks_list) == 0:
        return tasks_list, goal_list

    current_tasks_list = tasks_list

    tasks_list = []
    goal_list = []

    for subgoal in current_tasks_list:

        PLANNING_PROMPT = llm_prompt.compose_planning_prompt(rules)

        response = llm_utils.llm_chat(prompt="Current goal: " + subgoal, system_prompt=PLANNING_PROMPT, model="deepseek-chat")
        subgoals_list = []
        try: 
            llm_subgoals_list = ast.literal_eval(response)
            for new_subgoal in llm_subgoals_list:
                response = llm_utils.llm_chat(prompt=new_subgoal,system_prompt=llm_prompt.TRANS_PROMPT, model="deepseek-chat")
                if "None" not in response and response not in goal_list:
                    tasks_list.append(new_subgoal)
                    goal_list.append(response)

        except Exception as e:
            pass

    # print(tasks_list)
    # print(goal_list)

    return plan_aux(tasks_list, goal_list, current_step+1, num_step, rules)


if __name__ == "__main__":

    config = {"rules_path": os.path.join("temp_result", "rules.txt"),
              "tasks_list": ["collect an iron"],
              "planning_steps": 3,
              "save_plan": True,
              "save_goal_list": True,
              "save_plan_path": os.path.join("temp_result", "plan1.txt"),
              "save_goal_list_path": os.path.join("temp_result", "goal_list1.txt")
              }

    rules = open(config["rules_path"], 'r').read()
    tasks_list = config["tasks_list"]
    planning_steps = config["planning_steps"]
    
    tasks_list, goal_list = plan(tasks_list, planning_steps, rules)

    # print(tasks_list)
    # print(goal_list)

    if config["save_plan"]:

        with open(config["save_plan_path"], 'w') as f:
            f.write(str(tasks_list))
        print("plan successful saved at ", config["save_plan_path"])

    if config["save_goal_list"]:

        with open(config["save_goal_list_path"], 'w') as f:
            f.write(str(goal_list))
        print("goal list successful saved at ", config["save_goal_list_path"])

        print("LLM generated goals: ")
        print(goal_list)
