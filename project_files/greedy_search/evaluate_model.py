import os
import torch
import numpy as np
import pandas as pd
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

from cnn_transformer_model import CNNTransformer # our transformer model
from evaluate_model_helpers import *

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained CNNTransformer model on the copy task dataset.')
parser.add_argument('--model_path', type=str, default='trained_models/cnn_transformer_attempt1',
                    help='Path to the pretrained model directory (relative to the current working directory).')
parser.add_argument('--data_dir', type=str, default='../../data/hdf5_data_final',
                    help='Path to the dataset directory (relative to the current working directory).')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" for validation set, "test" for test set. '
                         'If "test", ground truth is not available.')
parser.add_argument('--csv_path', type=str, default='../../data/t15_copyTaskData_description.csv',
                    help='Path to the CSV file with metadata about the dataset (relative to the current working directory).')
parser.add_argument('--gpu_number', type=int, default=1,
                    help='GPU number to use for CNNTransformer model inference. Set to -1 to use CPU.')
args = parser.parse_args()

# paths to model and data directories
# Note: these paths are relative to the current working directory
model_path = args.model_path
data_dir = args.data_dir

# define evaluation type
eval_type = args.eval_type  # can be 'val' or 'test'. if 'test', ground truth is not available

# load csv file
b2txt_csv_df = pd.read_csv(args.csv_path)

# load model args
model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

# set up gpu device
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = f'cuda:{gpu_number}'
    device = torch.device(device)
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU number {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# define model
model = CNNTransformer(
    neural_dim=model_args['model']['n_input_features'],
    n_units=model_args['model']['n_units'],
    n_days=len(model_args['dataset']['sessions']),
    n_classes=model_args['dataset']['n_classes'],

    # conv config
    conv_channels=model_args['model']['conv_channels'],
    conv_kernel_sizes=model_args['model']['conv_kernel_sizes'],
    conv_strides=model_args['model']['conv_strides'],
    conv_residual=model_args['model']['conv_residual'],

    # transformer config
    enc_layers=model_args['model']['enc_layers'],
    dec_layers=model_args['model']['dec_layers'],
    n_heads=model_args['model']['n_heads'],
    dim_feedforward=model_args['model']['dim_feedforward'],
    trans_dropout=model_args['model']['trans_dropout'],
    input_dropout=model_args['model']['input_network']['input_layer_dropout'],
    activation=model_args['model']['activation'],
    max_len=model_args['model']['max_len'],
)

# load model weights
checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False)
# rename keys to not start with "module." (happens if model was saved with DataParallel)
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
model.load_state_dict(checkpoint['model_state_dict'])  

# add model to device
model.to(device) 

# set model to eval mode
model.eval()

# load data for each session
test_data = {}
total_test_trials = 0
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')

        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data

        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
print(f'Total number of {eval_type} trials: {total_test_trials}')
print()

# Special token IDs
SOS_IDX = 41
EOS_IDX = 42
PAD_IDX = 0

# put neural data through the pretrained model to get phoneme predictions
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():

        # they're not here anymore
        # data['logits'] = []

        data['pred_seq'] = []
        input_layer = model_args['dataset']['sessions'].index(session)

        for trial in range(len(data['neural_features'])):
            # get neural input for the trial
            neural_input = data['neural_features'][trial]

            # add batch dimension
            neural_input = np.expand_dims(neural_input, axis=0)

            # convert to torch tensor
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # smooth the data
            with torch.autocast(device_type="cuda", enabled=model_args['use_amp'], dtype=torch.bfloat16):
                neural_input = gauss_smooth(
                    inputs=neural_input,
                    device=device,
                    smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
                    smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
                    padding='valid',
                )

            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=model_args['use_amp'], dtype=torch.bfloat16):
                    # Encode neural data to memory
                    memory = model.encode(neural_input, torch.tensor([input_layer], device=device))

                    # Greedy decoding
                    generated_greedy = model.greedy_decode(memory, max_length=model_args['greedy_search']['max_decode_length'])
                    generated_beam = generated_greedy

            # Process greedy sequence
            raw_pred_greedy = generated_greedy[0][1:]  # Remove SOS
            pred_seq_greedy = []
            for token in raw_pred_greedy:
                if token == EOS_IDX: break
                if token == PAD_IDX: continue
                pred_seq_greedy.append(token.item())

            # Process beam search sequence
            raw_pred_beam = generated_beam[0][1:]  # Remove SOS
            pred_seq_beam = []
            for token in raw_pred_beam:
                if token == EOS_IDX: break
                if token == PAD_IDX: continue
                pred_seq_beam.append(token.item())

            data['pred_seq'].append(pred_seq_greedy)

            pbar.update(1)
pbar.close()

# convert predictions to phoneme sequences and print them out
for session, data in test_data.items():
    for trial in range(len(data['pred_seq'])):
        # Convert to phonemes
        pred_seq_greedy = [LOGIT_TO_PHONEME[p] for p in data['pred_seq'][trial]]

        # print out the predicted sequences
        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]
        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        if eval_type == 'val':
            sentence_label = data['sentence_label'][trial]
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]

            print(f'Sentence label:          {sentence_label}')
            print(f'True sequence:           {" ".join(true_seq)}')
        print(f'Predicted (greedy):      {" ".join(pred_seq_greedy)}')
        print()

# write predicted phoneme sequences to a csv file
output_file = os.path.join(model_path, f'CNNTransformer_GreedySearch_{eval_type}_predicted_phonemes_{time.strftime("%Y%m%d_%H%M%S")}.csv')
results_list = []
for session, data in test_data.items():
    for trial in range(len(data['pred_seq'])):
        pred_seq_greedy = ' '.join([LOGIT_TO_PHONEME[p] for p in data['pred_seq'][trial]])
        results_list.append({
            'session': session,
            'block': data['block_num'][trial],
            'trial': data['trial_num'][trial],
            'pred_phonemes_greedy': pred_seq_greedy,
        })
df_out = pd.DataFrame(results_list)
df_out.to_csv(output_file, index=False)
print(f"\nPredictions saved to: {output_file}")

"""

    Current problem: 
    We have the phonemes, but how do we convert them to sentences?
    The transformer-encoder -> n-gram -> OPT model pipeline worked, because the transformer-encoder outputted logits...    

"""
