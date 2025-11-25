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

from cnn_transformer_model import CNNTransformer # our transformer model with beam search
from evaluate_model_helpers import *

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained CNNTransformer model with beam search on the copy task dataset.')
parser.add_argument('--model_path', type=str, default='trained_models/cnn_transformer_beam',
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
parser.add_argument('--num_beams', type=int, default=5,
                    help='Number of beams for beam search decoding.')
parser.add_argument('--length_penalty', type=float, default=1.0,
                    help='Length penalty for beam search (>1 favors longer, <1 favors shorter).')
parser.add_argument('--use_beam_search', action='store_true', default=True,
                    help='Use beam search instead of greedy decoding.')
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

        data['logits'] = []
        data['pred_seq_greedy'] = []
        data['pred_seq_beam'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            # get neural input for the trial
            neural_input = data['neural_features'][trial]

            # add batch dimension
            neural_input = np.expand_dims(neural_input, axis=0)

            # convert to torch tensor
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # smooth the data
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
                    generated_greedy = model.greedy_decode(memory, max_length=100)
                    
                    # Beam search decoding
                    if args.use_beam_search:
                        generated_beam = model.beam_search_decode(
                            memory, 
                            num_beams=args.num_beams,
                            max_length=100,
                            length_penalty=args.length_penalty
                        )
                    else:
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

            data['pred_seq_greedy'].append(pred_seq_greedy)
            data['pred_seq_beam'].append(pred_seq_beam)

            pbar.update(1)
pbar.close()


# convert predictions to phoneme sequences and print them out
for session, data in test_data.items():
    for trial in range(len(data['pred_seq_beam'])):
        # Convert to phonemes
        pred_seq_greedy = [LOGIT_TO_PHONEME[p] for p in data['pred_seq_greedy'][trial]]
        pred_seq_beam = [LOGIT_TO_PHONEME[p] for p in data['pred_seq_beam'][trial]]
        
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
        print(f'Predicted (beam search): {" ".join(pred_seq_beam)}')
        print()


# language model inference via redis
# make sure that the standalone language model is running on the localhost redis ip
# see README.md for instructions on how to run the language model
r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()  # clear all streams in redis

# define redis streams for the remote language model
remote_lm_input_stream = 'remote_lm_input'
remote_lm_output_partial_stream = 'remote_lm_output_partial'
remote_lm_output_final_stream = 'remote_lm_output_final'

# set timestamps for last entries seen in the redis streams
remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_finalizing_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_updating_lastEntrySeen = get_current_redis_time_ms(r)

lm_results = {
    'session': [],
    'block': [],
    'trial': [],
    'true_sentence': [],
    'pred_sentence': [],
    'decoding_method': [],
}

print("\nNote: Language model integration requires the standalone LM server to be running.")
print("Skipping LM inference for now. Showing phoneme-level predictions above.")
print("\nTo run full LM inference, start the LM server and uncomment the LM inference code.")

# write predicted phoneme sequences to a csv file
output_file = os.path.join(model_path, f'CNNTransformer_BeamSearch_{eval_type}_predicted_phonemes_{time.strftime("%Y%m%d_%H%M%S")}.csv')
results_list = []
for session, data in test_data.items():
    for trial in range(len(data['pred_seq_beam'])):
        pred_seq_greedy = ' '.join([LOGIT_TO_PHONEME[p] for p in data['pred_seq_greedy'][trial]])
        pred_seq_beam = ' '.join([LOGIT_TO_PHONEME[p] for p in data['pred_seq_beam'][trial]])
        results_list.append({
            'session': session,
            'block': data['block_num'][trial],
            'trial': data['trial_num'][trial],
            'pred_phonemes_greedy': pred_seq_greedy,
            'pred_phonemes_beam': pred_seq_beam,
        })

df_out = pd.DataFrame(results_list)
df_out.to_csv(output_file, index=False)
print(f"\nPredictions saved to: {output_file}")
