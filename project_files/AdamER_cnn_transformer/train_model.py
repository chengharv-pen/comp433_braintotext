from omegaconf import OmegaConf
from cnn_transformer_trainer import BrainToText_Trainer

args = OmegaConf.load('cnn_transformer_args.yaml')
trainer = BrainToText_Trainer(args)
metrics = trainer.train()

