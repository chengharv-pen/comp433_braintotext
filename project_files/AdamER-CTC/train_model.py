from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import json
from datetime import datetime

args = OmegaConf.load('rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"./metrics/output_{timestamp}.json"

with open(filename, "w") as f:
    json.dump(metrics, f, indent=4)