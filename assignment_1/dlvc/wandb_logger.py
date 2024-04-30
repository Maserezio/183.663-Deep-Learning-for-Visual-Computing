import torch
import wandb
from typing import Dict

wandb.login(key="efd81210b4dc4c746b613e952e447cf548442a6f")

class WandBLogger:

    def __init__(self, enabled=True, 
                 model: torch.nn.modules=None, 
                 run_name: str=None,
                 config: Dict[str, any]=None) -> None:
        
        self.enabled = enabled
        self.config = config


        if self.enabled:
            wandb.init(entity="artur-ohanian-01",
                       project="dlvc",
                       group="artur-ohanian-01",
                       config = self.config)

            if run_name is None:
                # wandb.run.name = wandb.run.id
                wandb.run.name = "Vit"
            else:
                wandb.run.name = run_name  

            if model is not None:
                self.watch(model)         
            
    def watch(self, model, log_freq: int=1):
        wandb.watch(model, log="all", log_freq=log_freq)
            

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)
 

    def finish(self):
        if self.enabled:
            wandb.finish()
