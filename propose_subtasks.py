import llm_prompt
import llm_utils

rules = open("rules.txt", 'r').read()

subtasks = llm_utils.llm_chat(prompt = llm_prompt.compose_submodel_prompt(rules, "collet an iron"), system_prompt="", model="deepseek-chat")

print(subtasks)
