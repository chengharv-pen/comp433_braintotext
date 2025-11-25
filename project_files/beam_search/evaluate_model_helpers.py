import torch
import numpy as np
import h5py
import time
import re

from data_augmentations import gauss_smooth

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',
]

def _extract_transcription(input):
    endIdx = np.argwhere(input == 0)[0, 0]
    trans = ''
    for c in range(endIdx):
        trans += chr(input[c])
    return trans

def load_h5py_file(file_path, b2txt_csv_df):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
        'corpus': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            # match this trial up with the csv to get the corpus name
            year, month, day = session.split('.')[1:]
            date = f'{year}-{month}-{day}'
            row = b2txt_csv_df[(b2txt_csv_df['Date'] == date) & (b2txt_csv_df['Block number'] == block_num)]
            corpus_name = row['Corpus'].values[0]

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
            data['corpus'].append(corpus_name)
    return data

def rearrange_speech_logits_pt(logits):
    # original order is [BLANK, phonemes..., SIL]
    # rearrange so the order is [BLANK, SIL, phonemes...]
    logits = np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)
    return logits

# single decoding step function.
# smooths data and puts it through the model.
def runSingleDecodingStep(x, input_layer, model, model_args, device):

    # Use autocast for efficiency
    with torch.autocast(device_type = "cuda", enabled = model_args['use_amp'], dtype = torch.bfloat16):

        x = gauss_smooth(
            inputs = x, 
            device = device,
            smooth_kernel_std = model_args['dataset']['data_transforms']['smooth_kernel_std'],
            smooth_kernel_size = model_args['dataset']['data_transforms']['smooth_kernel_size'],
            padding = 'valid',
        )

        with torch.no_grad():
            logits, _ = model(
                x = x,
                day_idx = torch.tensor([input_layer], device=device),
                states = None, # no initial states
                return_state = True,
            )

    # convert logits from bfloat16 to float32
    logits = logits.float().cpu().numpy()

    # # original order is [BLANK, phonemes..., SIL]
    # # rearrange so the order is [BLANK, SIL, phonemes...]
    # logits = rearrange_speech_logits_pt(logits)

    return logits

def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])

    return sentence

def get_current_redis_time_ms(redis_conn):
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)


######### language model helper functions ##########

def reset_remote_language_model(
        r,
        remote_lm_done_resetting_lastEntrySeen,
    ):
    
    r.xadd('remote_lm_reset', {'done': 0})
    time.sleep(0.001)
    # print('Resetting remote language model before continuing...')
    remote_lm_done_resetting = []
    while len(remote_lm_done_resetting) == 0:
        remote_lm_done_resetting = r.xread(
            {'remote_lm_done_resetting': remote_lm_done_resetting_lastEntrySeen},
            count=1,
            block=10000,
        )
        if len(remote_lm_done_resetting) == 0:
            print(f'Still waiting for remote lm reset from ts {remote_lm_done_resetting_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_done_resetting[0][1]:
        remote_lm_done_resetting_lastEntrySeen = entry_id
        # print('Remote language model reset.')

    return remote_lm_done_resetting_lastEntrySeen


def update_remote_lm_params(
        r,
        remote_lm_done_updating_lastEntrySeen,
        acoustic_scale=0.35,
        blank_penalty=90.0,
        alpha=0.55,
    ):
    
    # update remote lm params
    entry_dict = {
        # 'max_active': max_active,
        # 'min_active': min_active,
        # 'beam': beam,
        # 'lattice_beam': lattice_beam,
        'acoustic_scale': acoustic_scale,
        # 'ctc_blank_skip_threshold': ctc_blank_skip_threshold,
        # 'length_penalty': length_penalty,
        # 'nbest': nbest,
        'blank_penalty': blank_penalty,
        'alpha': alpha,
        # 'do_opt': do_opt,
        # 'rescore': rescore,
        # 'top_candidates_to_augment': top_candidates_to_augment,
        # 'score_penalty_percent': score_penalty_percent,
        # 'specific_word_bias': specific_word_bias,
    }

    r.xadd('remote_lm_update_params', entry_dict)
    time.sleep(0.001)
    remote_lm_done_updating = []
    while len(remote_lm_done_updating) == 0:
        remote_lm_done_updating = r.xread(
            {'remote_lm_done_updating_params': remote_lm_done_updating_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_done_updating) == 0:
            print(f'Still waiting for remote lm to update parameters from ts {remote_lm_done_updating_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_done_updating[0][1]:
        remote_lm_done_updating_lastEntrySeen = entry_id
        # print('Remote language model params updated.')

    return remote_lm_done_updating_lastEntrySeen


def finalize_trial(
        r,
        remote_lm_done_finalizing_lastEntrySeen,
    ):

    r.xadd('remote_lm_finalize', {'done': 0})
    time.sleep(0.001)
    remote_lm_done_finalizing = []
    while len(remote_lm_done_finalizing) == 0:
        remote_lm_done_finalizing = r.xread(
            {'remote_lm_done_finalizing': remote_lm_done_finalizing_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_done_finalizing) == 0:
            print(f'Still waiting for remote lm to finalize from ts {remote_lm_done_finalizing_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_done_finalizing[0][1]:
        remote_lm_done_finalizing_lastEntrySeen = entry_id

    return remote_lm_done_finalizing_lastEntrySeen
