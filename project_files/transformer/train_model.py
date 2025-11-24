from omegaconf import OmegaConf
from transformer_trainer import BrainToText_Trainer
import json
from datetime import datetime

args = OmegaConf.load('transformer_args.yaml')
trainer = BrainToText_Trainer(args)
metrics = trainer.train()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"./metrics/output_{timestamp}.json"

with open(filename, "w") as f:
    json.dump(metrics, f, indent=4)