import yaml
import os


class PromptManager():
    def __init__(self,file_path=r"src\config\prompts.yaml"):
        with open(file_path,'r') as f:
            self.prompts=yaml.safe_load(f)

    def get_prompt(self,node_name):
        node_data= self.prompts.get(node_name,{})
        return node_data.get('system_prompt')