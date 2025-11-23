import os
from typing import Iterator, List, Sequence

import h5py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def _decode_transcription(raw_transcription: Sequence[int]) -> str:
    chars: List[str] = []
    for value in raw_transcription:
        if value == 0:
            break
        chars.append(chr(int(value)))
    return ''.join(chars)


def _iter_hdf5_files(dataset_dir: str, sessions: Sequence[str], splits: Sequence[str]) -> Iterator[str]:
    for session in sessions:
        for split in splits:
            filename = f"data_{split}.hdf5"
            candidate = os.path.join(dataset_dir, session, filename)
            if os.path.exists(candidate):
                yield candidate


class BPETokenizerManager:
    """Handles training/loading of a Byte-Pair Encoding Tokenizer."""

    def __init__(self, tokenizer_cfg: dict, dataset_cfg: dict, logger=None):
        self.cfg = tokenizer_cfg
        self.dataset_cfg = dataset_cfg
        self.logger = logger
        self.tokenizer_path = self.cfg.get('save_path')
        if self.tokenizer_path is None:
            raise ValueError("tokenizer.save_path must be specified when tokenizer is enabled")
        self.lowercase = self.cfg.get('lowercase', True)
        self.special_tokens = self.cfg.get('special_tokens', {})
        self.train_splits = self.cfg.get('train_on_splits', ['train'])
        self.force_retrain = self.cfg.get('force_retrain', False)
        self.vocab_size = self.cfg.get('vocab_size', 512)
        self.min_frequency = self.cfg.get('min_frequency', 2)
        self.max_training_examples = self.cfg.get('max_training_examples')
        self._tokenizer = None

    def get_tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer

        if os.path.exists(self.tokenizer_path) and not self.force_retrain:
            if self.logger:
                self.logger.info(f"Loading BPE tokenizer from {self.tokenizer_path}")
            self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
            return self._tokenizer

        if self.logger:
            self.logger.info("Training new BPE tokenizer (file not found or retrain requested)...")

        self._tokenizer = self._train_and_save()
        return self._tokenizer

    def _train_and_save(self) -> Tokenizer:
        os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)

        unk_token = self.special_tokens.get('unk', '<unk>')
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self._special_tokens_list(),
        )

        iterator = self._transcription_iterator()
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        tokenizer.save(self.tokenizer_path)

        if self.logger:
            self.logger.info(
                f"Saved BPE tokenizer to {self.tokenizer_path} (vocab size: {tokenizer.get_vocab_size()})"
            )

        return tokenizer

    def _special_tokens_list(self) -> List[str]:
        seen = set()
        ordered = []
        for key in ['pad', 'bos', 'eos', 'unk']:
            token = self.special_tokens.get(key)
            if token and token not in seen:
                ordered.append(token)
                seen.add(token)
        for token in self.special_tokens.values():
            if token not in seen:
                ordered.append(token)
                seen.add(token)
        return ordered

    def _transcription_iterator(self) -> Iterator[str]:
        count = 0
        for file_path in _iter_hdf5_files(
            self.dataset_cfg['dataset_dir'],
            self.dataset_cfg['sessions'],
            self.train_splits,
        ):
            for text in self._texts_from_file(file_path):
                yield text
                count += 1
                if self.max_training_examples and count >= self.max_training_examples:
                    if self.logger:
                        self.logger.info("Reached max_training_examples while training tokenizer")
                    return

    def _texts_from_file(self, file_path: str) -> Iterator[str]:
        if self.logger:
            self.logger.info(f"Collecting transcriptions from {file_path}")
        with h5py.File(file_path, 'r') as handle:
            for trial_key in handle.keys():
                raw_transcription = handle[trial_key]['transcription'][:]
                text = _decode_transcription(raw_transcription)
                if not text:
                    continue
                if self.lowercase:
                    text = text.lower()
                yield text