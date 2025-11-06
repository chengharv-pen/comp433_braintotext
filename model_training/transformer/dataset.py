import os
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class SimpleBrainToTextDataset(Dataset):
    '''
    Simple Dataset for brain-to-text data without day grouping.
    Loads all trials from all sessions into a flat list.
    Returns individual trials, not batches.
    '''

    def __init__(
        self,
        file_paths,
        split='train',
        test_percentage=0.1,
        seed=-1,
        bad_trials_dict=None,
        feature_subset=None,
        use_augmentation=False,
        augmentation_params=None
    ):
        '''
        file_paths: (list) - list of paths to HDF5 files
        split: (str) - 'train' or 'test'
        test_percentage: (float) - percentage for test split
        seed: (int) - random seed
        bad_trials_dict: (dict) - bad trials to exclude
        feature_subset: (list) - subset of features
        use_augmentation: (bool) - whether to use data augmentation
        augmentation_params: (dict) - parameters for augmentation
        '''

        if seed != -1:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.split = split
        self.use_augmentation = use_augmentation
        self.augmentation_params = augmentation_params or {}
        self.feature_subset = feature_subset

        # Load all trials
        self.trials = []
        for path in file_paths:
            if os.path.exists(path):
                with h5py.File(path, 'r') as f:
                    session = [s for s in path.split('/') if (s.startswith('t15.20') or s.startswith('t12.20'))][0]
                    num_trials = len(list(f.keys()))
                    for t in range(num_trials):
                        key = f'trial_{t:04d}'

                        block_num = f[key].attrs['block_num']
                        trial_num = f[key].attrs['trial_num']

                        if bad_trials_dict and session in bad_trials_dict:
                            if str(block_num) in bad_trials_dict[session] and trial_num in bad_trials_dict[session][str(block_num)]:
                                continue

                        trial_data = {
                            'input_features': torch.from_numpy(f[key]['input_features'][:]),
                            'seq_class_ids': torch.from_numpy(f[key]['seq_class_ids'][:]),
                            'transcription': torch.from_numpy(f[key]['transcription'][:]),
                            'n_time_steps': f[key].attrs['n_time_steps'],
                            'seq_len': f[key].attrs['seq_len'],
                            'block_num': f[key].attrs['block_num'],
                            'trial_num': f[key].attrs['trial_num'],
                            'session_path': path
                        }
                        self.trials.append(trial_data)

        # Split into train/test
        n_total = len(self.trials)
        n_test = int(n_total * test_percentage)
        indices = np.random.permutation(n_total)

        if split == 'train':
            self.trials = [self.trials[i] for i in indices[:-n_test]]
        else:
            self.trials = [self.trials[i] for i in indices[-n_test:]]

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]

        input_features = trial['input_features']
        if self.feature_subset:
            input_features = input_features[:, self.feature_subset]

        seq_class_ids = trial['seq_class_ids']

        # Add SOS and EOS tokens
        # Assuming phonemes are 1-40, 0 is pad, 41 is SOS, 42 is EOS
        sos_token = 41
        eos_token = 42
        seq_class_ids = torch.cat([torch.tensor([sos_token]), seq_class_ids, torch.tensor([eos_token])])

        # Apply augmentation if enabled
        if self.use_augmentation and self.split == 'train':
            input_features = self._augment(input_features)

        return {
            'input_features': input_features,  # [time_steps, features]
            'seq_class_ids': seq_class_ids,  # [seq_len + 2] with SOS and EOS
            'transcription': trial['transcription'],  # [seq_len]
            'n_time_steps': trial['n_time_steps'],
            'seq_len': trial['seq_len'] + 2  # Include SOS and EOS
        }

    def _augment(self, features):
        '''Apply data augmentations'''
        # Simple augmentations - can be expanded
        if 'white_noise_std' in self.augmentation_params and self.augmentation_params['white_noise_std'] > 0:
            noise = torch.randn_like(features) * self.augmentation_params['white_noise_std']
            features = features + noise

        if 'constant_offset_std' in self.augmentation_params and self.augmentation_params['constant_offset_std'] > 0:
            offset = torch.randn(features.shape[-1]) * self.augmentation_params['constant_offset_std']
            features = features + offset

        return features

def collate_fn(batch):
    '''Collate function for DataLoader'''
    input_features = [item['input_features'] for item in batch]
    seq_class_ids = [item['seq_class_ids'] for item in batch]
    transcriptions = [item['transcription'] for item in batch]
    n_time_steps = torch.tensor([item['n_time_steps'] for item in batch])
    seq_lens = torch.tensor([item['seq_len'] for item in batch])

    # Pad sequences
    input_features = pad_sequence(input_features, batch_first=True, padding_value=0)
    seq_class_ids = pad_sequence(seq_class_ids, batch_first=True, padding_value=0)

    return {
        'input_features': input_features,
        'seq_class_ids': seq_class_ids,
        'transcriptions': transcriptions,
        'n_time_steps': n_time_steps,
        'seq_lens': seq_lens
    }