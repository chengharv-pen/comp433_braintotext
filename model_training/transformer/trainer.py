import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
from .dataset import SimpleBrainToTextDataset, collate_fn
from .model import BasicTransformer
import torchaudio.functional as F

class TransformerTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create datasets
        file_paths = [os.path.join(config['dataset_dir'], session, 'data_train.hdf5')
                     for session in config['sessions']]

        self.train_dataset = SimpleBrainToTextDataset(
            file_paths=file_paths,
            split='train',
            test_percentage=config['test_percentage'],
            seed=config['seed'],
            use_augmentation=config['use_augmentation'],
            augmentation_params=config.get('augmentation_params', {})
        )

        self.val_dataset = SimpleBrainToTextDataset(
            file_paths=file_paths,
            split='test',
            test_percentage=config['test_percentage'],
            seed=config['seed'],
            use_augmentation=False  # No augmentation for validation
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )

        # Create model
        self.model = BasicTransformer(
            src_d_model=config['src_d_model'],
            tgt_vocab=config['tgt_vocab'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            n_encoder_layers=config['n_encoder_layers'],
            n_decoder_layers=config['n_decoder_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout']
        ).to(self.device)

        # Use torch.compile for speed
        self.model = torch.compile(self.model)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-4))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is pad

        # Scaler for mixed precision
        self.scaler = torch.amp.GradScaler() if config.get('use_amp', True) else None

        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def train_epoch(self, log_losses):
        self.model.train()
        total_loss = 0
        batch_losses = []

        for i, batch in enumerate(self.train_loader):
            src = batch['input_features'].to(self.device)  # [batch, src_len, features]
            tgt = batch['seq_class_ids'].to(self.device)   # [batch, tgt_len]

            # Create masks
            src_padding_mask = (src.sum(dim=-1) == 0)  # [batch, src_len]
            tgt_padding_mask = (tgt == 0)  # [batch, tgt_len]

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            tgt_input_padding_mask = tgt_padding_mask[:, :-1]

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = self.model(src, tgt_input, src_padding_mask, tgt_input_padding_mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_target.view(-1))

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(src, tgt_input, src_padding_mask, tgt_input_padding_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_target.view(-1))
                loss.backward()
                self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)

            if log_losses and (i + 1) % self.config.get('log_every', 10) == 0:
                print(f"Batch {i+1}: Loss {batch_loss:.4f}")

        return total_loss / len(self.train_loader), batch_losses

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_per = 0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['input_features'].to(self.device)
                tgt = batch['seq_class_ids'].to(self.device)

                src_padding_mask = (src.sum(dim=-1) == 0)
                tgt_padding_mask = (tgt == 0)

                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]
                tgt_input_padding_mask = tgt_padding_mask[:, :-1]

                logits = self.model(src, tgt_input, src_padding_mask, tgt_input_padding_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_target.view(-1))
                total_loss += loss.item()

                # Calculate PER (simplified)
                pred_seqs = []
                true_seqs = []

                for i in range(len(batch['seq_lens'])):
                    # Greedy decode
                    pred = self.model.generate(src[i:i+1], max_new_tokens=100, sos_id=self.config['sos_id'], eos_id=self.config['eos_id'])
                    pred = pred[0].cpu().numpy()
                    # Remove SOS and EOS and pads
                    pred = pred[1:]  # Remove SOS
                    if self.config['eos_id'] in pred:
                        eos_idx = np.nonzero(pred == self.config['eos_id'])[0][0]
                        pred = pred[:eos_idx]
                    pred = pred[pred != 0]  # Remove pads if any
                    pred_seqs.append(pred)

                    # True seq: original without SOS/EOS
                    true = batch['seq_class_ids'][i].cpu().numpy()
                    true = true[1:-1]  # Remove SOS and EOS
                    true = true[true != 0]
                    true_seqs.append(true)

                    per = F.edit_distance(pred, true)
                    total_per += per

                count += len(batch['seq_lens'])

        avg_loss = total_loss / len(self.val_loader)
        avg_per = total_per / count if count > 0 else 0

        return avg_loss, avg_per

    def train(self, max_time_seconds=None, log_losses=True):
        start_time = time.time()
        epoch = 0
        all_train_losses = []
        all_val_losses = []
        all_val_pers = []

        while True:
            epoch_start = time.time()
            train_loss, batch_losses = self.train_epoch(log_losses)
            val_loss, val_per = self.validate()

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time

            all_train_losses.extend(batch_losses)
            all_val_losses.append(val_loss)
            all_val_pers.append(val_per)

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PER: {val_per:.4f}, Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s")

            if val_per < getattr(self, 'best_per', float('inf')):
                self.best_per = val_per
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
                print("Saved best model")

            epoch += 1

            if max_time_seconds and total_time > max_time_seconds:
                break

        # Save losses for plotting
        with open('training_losses.json', 'w') as f:
            json.dump({
                'train_losses': all_train_losses,
                'val_losses': all_val_losses,
                'val_pers': all_val_pers
            }, f)

        print(f"Training complete. Best PER: {self.best_per:.4f}")
        return all_train_losses, all_val_losses, all_val_pers