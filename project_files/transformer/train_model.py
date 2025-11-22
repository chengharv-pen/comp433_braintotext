from omegaconf import OmegaConf
from transformer_trainer import BrainToText_Trainer

args = OmegaConf.load('transformer_args.yaml')
trainer = BrainToText_Trainer(args)
metrics = trainer.train()