import torch 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle

# Add model_training to path to import dataset and data_augmentations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_training')))

from dataset import BrainToTextDataset, train_test_split_indicies
from data_augmentations import gauss_smooth

import torchaudio.functional as F # for edit distance
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high') 
torch.backends.cudnn.deterministic = True 
torch._dynamo.config.cache_size_limit = 64

from transformer_model import TransformerModel

class BrainToTextDecoderTrainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder using a Transformer
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.criterion = None 

        self.best_val_PER = torch.inf 
        self.best_val_loss = torch.inf 

        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 

        self.transform_args = self.args['dataset']['data_transforms']

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=True)

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']: 
            os.makedirs(self.args['checkpoint_dir'], exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]: 
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device pytorch will use 
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                gpu_num = 0
            self.device = torch.device(f"cuda:{gpu_num}")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        # Set seed if provided 
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        # Initialize the model 
        self.model = TransformerModel(
            neural_dim = self.args['model']['n_input_features'],
            n_classes  = self.args['dataset']['n_classes'],
            n_days = len(self.args['dataset']['sessions']),
            d_model = self.args['model']['d_model'],
            nhead = self.args['model']['nhead'],
            num_encoder_layers = self.args['model']['num_encoder_layers'],
            num_decoder_layers = self.args['model']['num_decoder_layers'],
            dim_feedforward = self.args['model']['dim_feedforward'],
            dropout = self.args['model']['dropout'],
            input_dropout = self.args['model']['input_network']['input_layer_dropout'],
            patch_size = self.args['model']['patch_size'],
            patch_stride = self.args['model']['patch_stride'],
        )

        # Call torch.compile to speed up training
        self.logger.info("Using torch.compile")
        self.model = torch.compile(self.model)

        self.logger.info(f"Initialized Transformer decoding model")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        # Create datasets and dataloaders
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]

        # Split trials into train and test sets
        train_trials, _ = train_test_split_indicies(
            file_paths = train_file_paths, 
            test_percentage = 0,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )
        _, val_trials = train_test_split_indicies(
            file_paths = val_file_paths, 
            test_percentage = 1,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )

        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            json.dump({'train' : train_trials, 'val': val_trials}, f)

        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None: 
            feature_subset = self.args['dataset']['feature_subset']
            
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials,
            split = 'train',
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, 
            shuffle = self.args['dataset']['loader_shuffle'],
            num_workers = self.args['dataset']['num_dataloader_workers'],
            pin_memory = True 
        )

        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, 
            split = 'test',
            days_per_batch = None,
            n_batches = None,
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset   
            )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, 
            shuffle = False, 
            num_workers = 0,
            pin_memory = True 
        )

        self.logger.info("Successfully initialized datasets")

        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min'] / self.args['lr_max'],
                total_iters = self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")
        
        # CrossEntropyLoss for Transformer
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        self.model.to(self.device)

    def create_optimizer(self):
        bias_params = [p for name, p in self.model.named_parameters() if 'bias' in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'bias' not in name]

        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else: 
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
            
        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = True
        )

        return optim 

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps), 
                lambda step: lr_lambda(step, lr_min_day / lr_max_day, lr_decay_steps_day, lr_warmup_steps_day), 
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps), 
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps), 
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps), 
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")
        
        return LambdaLR(optim, lr_lambdas, -1)
        
    def load_model_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, weights_only = False) 

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER'] 
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf

        self.model.to(self.device)
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss):
        checkpoint = {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.learning_rate_scheduler.state_dict(),
            'val_PER' : PER,
            'val_loss' : loss
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info("Saved model to checkpoint: " + save_path)

        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    def transform_data(self, features, n_time_steps, mode = 'train'):
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        if mode == 'train':
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']
                features = torch.matmul(features, warp_mat)

            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
            
        return features, n_time_steps

    def prepare_targets(self, labels, phone_seq_lens):
        batch_size = labels.size(0)
        max_len = labels.size(1)
        
        SOS = 42
        EOS = 43
        PAD = 0
        
        tgt_input = torch.full((batch_size, max_len + 1), PAD, dtype=torch.long, device=self.device)
        tgt_output = torch.full((batch_size, max_len + 1), PAD, dtype=torch.long, device=self.device)
        
        tgt_input[:, 0] = SOS
        
        for i in range(batch_size):
            length = phone_seq_lens[i]
            if length > 0:
                tgt_input[i, 1:length+1] = labels[i, :length]
                tgt_output[i, :length] = labels[i, :length]
                tgt_output[i, length] = EOS
            else:
                tgt_output[i, 0] = EOS
                
        return tgt_input, tgt_output

    def train(self):
        self.model.train()

        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)
        early_stopping_val_steps = self.args['early_stopping_val_steps']

        train_start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            
            self.model.train()
            self.optimizer.zero_grad()
            
            start_time = time.time() 

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):

                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')

                tgt_input, tgt_output = self.prepare_targets(labels, phone_seq_lens)
                
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=self.device)
                tgt_key_padding_mask = (tgt_input == 0)

                logits = self.model(features, day_indicies, tgt_input, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

                # Flatten logits and targets for CrossEntropyLoss
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
            
            loss.backward()

            if self.args['grad_norm_clip_value'] > 0: 
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               max_norm = self.args['grad_norm_clip_value'],
                                               error_if_nonfinite = True,
                                               foreach = True
                                               )

            self.optimizer.step()
            self.learning_rate_scheduler.step()
            
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            if i % self.args['batches_per_train_log'] == 0:
                self.logger.info(f'Train batch {i}: ' +
                        f'loss: {(loss.detach().item()):.2f} ' +
                        f'grad norm: {grad_norm:.2f} '
                        f'time: {train_step_duration:.3f}')

            if i % self.args['batches_per_val_step'] == 0 or i == ((self.args['num_training_batches'] - 1)):
                self.logger.info(f"Running test after training batch: {i}")
                
                start_time = time.time()
                val_metrics = self.validation(loader = self.val_loader, return_logits = self.args['save_val_logits'], return_data = self.args['save_val_data'])
                val_step_duration = time.time() - start_time

                self.logger.info(f'Val batch {i}: ' +
                        f'PER (avg): {val_metrics["avg_PER"]:.4f} ' +
                        f'Loss (avg): {val_metrics["avg_loss"]:.4f} ' +
                        f'time: {val_step_duration:.3f}')
                
                if self.args['log_individual_day_val_PER']:
                    for day in val_metrics['day_PERs'].keys():
                        self.logger.info(f"{self.args['dataset']['sessions'][day]} val PER: {val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']:0.4f}")

                val_PERs.append(val_metrics['avg_PER'])
                val_losses.append(val_metrics['avg_loss'])
                val_results.append(val_metrics)

                new_best = False
                if val_metrics['avg_PER'] < self.best_val_PER:
                    self.logger.info(f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                    self.best_val_PER = val_metrics['avg_PER']
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True
                elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss): 
                    self.logger.info(f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True

                if new_best:
                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/best_checkpoint', self.best_val_PER, self.best_val_loss)

                    if self.args['save_val_metrics']:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                            pickle.dump(val_metrics, f) 

                    val_steps_since_improvement = 0
                    
                else:
                    val_steps_since_improvement +=1

                if self.args['save_all_val_steps']:
                    self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}', val_metrics['avg_PER'])

                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(f'Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}')
                    break
                
        training_duration = time.time() - train_start_time

        self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        if self.args['save_final_model']:
            self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1])

        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses 
        train_stats['val_PERs'] = val_PERs
        train_stats['val_metrics'] = val_results

        return train_stats

    def validation(self, loader, return_logits = False, return_data = False):
        self.model.eval()

        metrics = {}
        
        if return_logits: 
            metrics['logits'] = []
            metrics['n_time_steps'] = []

        if return_data: 
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []
        metrics['true_seq'] = []
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []

        total_edit_distance = 0
        total_seq_length = 0

        day_per = {}
        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1: 
                day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for i, batch in enumerate(loader):        

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            day = day_indicies[0].item()
            if self.args['dataset']['dataset_probability_val'][day] == 0: 
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day}")
                continue
            
            with torch.no_grad():

                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    tgt_input, tgt_output = self.prepare_targets(labels, phone_seq_lens)
                    
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=self.device)
                    tgt_key_padding_mask = (tgt_input == 0)

                    logits = self.model(features, day_indicies, tgt_input, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    
                    loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))

                metrics['losses'].append(loss.cpu().detach().numpy())

                # Greedy decoding for validation
                # For Transformer, we need to generate token by token
                # This is slow, but necessary for PER calculation
                # We can limit max length
                
                batch_edit_distance = 0 
                decoded_seqs = []
                
                # Simple greedy decoding
                # Start with SOS
                SOS = 42
                EOS = 43
                
                curr_tgt = torch.full((features.size(0), 1), SOS, dtype=torch.long, device=self.device)
                
                # Max length for generation
                max_gen_len = labels.size(1) + 10
                
                finished = torch.zeros(features.size(0), dtype=torch.bool, device=self.device)
                
                # Cache encoder output? nn.Transformer doesn't expose encoder output easily in forward
                # But we can just run forward repeatedly (inefficient but simple)
                # Or we can modify model to return encoder output and take it as input
                # For now, let's just run forward repeatedly. It's validation, so maybe okay.
                
                # Actually, for PER calculation, we really need the decoded sequence.
                # Let's do a simplified decoding: just argmax at each step? No, it's autoregressive.
                
                for t in range(max_gen_len):
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(curr_tgt.size(1), device=self.device)
                    # We don't need padding mask for curr_tgt as it has no padding yet (or we can ignore it)
                    
                    out = self.model(features, day_indicies, curr_tgt, tgt_mask=tgt_mask)
                    # out: [batch, seq_len, vocab]
                    
                    next_token_logits = out[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    
                    curr_tgt = torch.cat([curr_tgt, next_token.unsqueeze(1)], dim=1)
                    
                    # Check for EOS
                    finished |= (next_token == EOS)
                    if finished.all():
                        break
                
                # Process decoded sequences
                for iterIdx in range(features.size(0)):
                    # Remove SOS (first token)
                    pred = curr_tgt[iterIdx, 1:].cpu().numpy()
                    
                    # Cut at EOS
                    try:
                        eos_idx = np.where(pred == EOS)[0][0]
                        pred = pred[:eos_idx]
                    except IndexError:
                        pass
                        
                    # Remove padding if any (shouldn't be if we stopped at EOS)
                    pred = np.array([p for p in pred if p != 0])

                    trueSeq = np.array(
                        labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                    )
            
                    batch_edit_distance += F.edit_distance(pred, trueSeq)

                    decoded_seqs.append(pred)

            day = batch['day_indicies'][0].item()
                
            day_per[day]['total_edit_distance'] += batch_edit_distance
            day_per[day]['total_seq_length'] += torch.sum(phone_seq_lens).item()


            total_edit_distance += batch_edit_distance
            total_seq_length += torch.sum(phone_seq_lens)

            if return_logits: 
                metrics['logits'].append(logits.cpu().float().numpy()) 
                metrics['n_time_steps'].append(n_time_steps.cpu().numpy()) # Approximate

            if return_data: 
                metrics['input_features'].append(batch['input_features'].cpu().numpy()) 

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

        avg_PER = total_edit_distance / total_seq_length

        metrics['day_PERs'] = day_per
        metrics['avg_PER'] = avg_PER.item()
        metrics['avg_loss'] = np.mean(metrics['losses'])

        return metrics
