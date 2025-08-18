import llm_prompt
import llm_utils

config = {"rules_path": "rules.txt", 
          "plan_path": "plan.txt", 
          "save": False, 
          "save_path": "./submodels.json",
          "goal": "collect an iron"}

rules = open(config["rules_path"], 'r').read()
plan = open(config["plan_path"], 'r').read()

subtasks = llm_utils.llm_chat(prompt = llm_prompt.compose_submodel_prompt(rules, plan, config["goal"]), system_prompt="", model="deepseek-chat")

if config["save"]:
    with open(config["save_path"], 'w') as f:
        f.write(subtasks)

print(subtasks)
