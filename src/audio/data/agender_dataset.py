import sys

sys.path.append('src')

import os
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torchvision

from torch.utils.data import Dataset

from configs.unimodal_config import data_config as conf

from common.data.common import load_data, save_data, slice_audio, find_intersections, read_audio, generate_features_file_name, gender_label_to_int, mask_label_to_int
from common.data.data_preprocessors import BaseDataPreprocessor


class AGenderDataset(Dataset):
    def __init__(self, data_root: str, labels_metadata: pd.DataFrame, 
                 features_root: str, features_file_name: str, 
                 corpus_name: str, gender_num_classes: int,
                 vad_metadata: dict[list] = None,
                 sr: int = 16000, win_max_length: int = 4, win_shift: int = 2, win_min_length: int = 0, 
                 transform: torchvision.transforms.transforms.Compose = None,
                 data_preprocessor: BaseDataPreprocessor = None) -> None:
        """AGender dataset
        Preprocesses labels and data

        Args:
            data_root (str): Audio root dir
            labels_metadata (pd.DataFrame): Pandas labels
            features_root (str): File path for fast load features
            features_file_name (str): File name for fast load features
            corpus_name (str): Name of corpus
            gender_num_classes (int): Number of gender classes
            vad_metadata (dict[list], optional): VAD information. Defaults to None.
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
            win_shift (int, optional): Shift length of window. Defaults to 2.
            win_min_length (int, optional): Min length of window. Defaults to 0.
            transform (torchvision.transforms.transforms.Compose, optional): Augmentation methods. Defaults to None.
            data_preprocessor (BaseDataProcessor, optional): Data preprocessor. Defaults to None.
        """
        self.data_root = data_root
        self.labels_metadata = labels_metadata
        self.vad_metadata = vad_metadata
        
        self.sr = sr
        self.win_max_length = win_max_length
        self.win_shift = win_shift
        self.win_min_length = win_min_length
        
        self.transform = transform
        self.data_preprocessor = data_preprocessor
        self.corpus_name = corpus_name
        self.gender_num_classes = gender_num_classes

        partial_features_file_name = generate_features_file_name(vad_metadata=self.vad_metadata,  
                                                                 win_max_length=self.win_max_length, 
                                                                 win_shift=self.win_shift,
                                                                 win_min_length=self.win_min_length)

        self.full_features_path = os.path.join(features_root, '{}_{}'.format(features_file_name, partial_features_file_name))
        full_features_file_name = os.path.join(features_root, '{}_{}_stats.pickle'.format(features_file_name, partial_features_file_name))

        if not os.path.exists(self.full_features_path):
            os.makedirs(self.full_features_path)
        
        self.info = load_data(full_features_file_name)

        if not self.info:
            self.prepare_data()
            save_data(self.info, full_features_file_name)
            
        self.info, self.stats = self.filter_samples(self.info, self.gender_num_classes)

    def prepare_data(self) -> None:
        """Prepares data
        - Reads audio
        - Slices audio
        - Finds windows intersections according to VAD
        - Saves features
        - Calculates label statistics
        """
        self.info = []

        for sample in tqdm(self.labels_metadata.to_dict('records')):
            sample_filename = sample['audio_file_path'].replace('.mp3', '.wav').replace('.m4a', '.wav')
            sample_fp = os.path.join(self.data_root, sample_filename)
            
            sample_gen = sample['gender']
            sample_age = sample['age']
            sample_mask = 'No mask'
            
            full_wave = read_audio(sample_fp, self.sr)

            audio_windows = slice_audio(start_time=0, end_time=int(len(full_wave)),
                                        win_max_length=int(self.win_max_length * self.sr), 
                                        win_shift=int(self.win_shift * self.sr), 
                                        win_min_length=int(self.win_min_length * self.sr))
            
            if not audio_windows:
                continue
            
            if self.vad_metadata:
                vad_info = self.vad_metadata[sample_filename]
                intersections = find_intersections(x=audio_windows, y=vad_info, min_length=int(self.win_min_length * self.sr))
            else:
                intersections = audio_windows

            for w_idx, window in enumerate(intersections):
                wave = full_wave[window['start']: window['end']].clone()
                
                self.info.append({
                    'fp': sample_fp,
                    'fn': sample_filename,
                    'w_idx': w_idx,
                    'start': window['start'],
                    'end': window['end'],
                    'gen': sample_gen,
                    'age': sample_age,
                    'mask': sample_mask,
                })
                
                save_data(wave, os.path.join(self.full_features_path, 
                                             sample_filename.replace('.wav', '_{0}.dat'.format(w_idx))))

    def filter_samples(self, info: list[dict], gender_num_classes: int) -> tuple[list[dict], dict]:
        """Filters samples according to number of gender classes and computes statistics for grouping of samples

        Args:
            info (list[dict]): List of samples with all information
            gender_num_classes (int): Number of gender classes

        Returns:
            tuple[list[dict], dict]: Filtered info dictionary and statistics for all samples
        """
        stats = {
            'fns': {},
            'majority_class': {
                'gen': 0,
                'age': 0,
                'mask': 0,
            },
            'counts': {
                'gen': np.asarray([0 for i in range(0, gender_num_classes)]),
                'age': 0,
                'mask': np.asarray([0 for i in range(0, 6)]),
            },
        }
        
        new_info = []
        for sample_info in info:
            if ('child' in sample_info['gen']) and (gender_num_classes < 3):
                continue
            
            sample_gen = gender_label_to_int(sample_info['gen'], self.gender_num_classes)
            sample_age = int(sample_info['age']) / 100.0
            sample_mask = mask_label_to_int(sample_info['mask'])
            
            new_info.append({
                'fp': sample_info['fp'],
                'fn': sample_info['fn'],
                'w_idx': sample_info['w_idx'],
                'start': sample_info['start'],
                'end': sample_info['end'],
                'gen': sample_gen,
                'age': sample_age,
                'mask': sample_mask,
            })
            
            stats['fns'][sample_info['fn']] = {
                'gen': sample_gen,
                'age': sample_age,
                'mask': sample_mask
            }
            
            stats['counts']['gen'][sample_gen] += 1
            stats['counts']['age'] += sample_age
            stats['counts']['mask'][sample_mask] += 1
            
        stats['majority_class']['gen'] = np.argmax(stats['counts']['gen'])
        stats['majority_class']['age'] = stats['counts']['age'] / sum(stats['counts']['gen'])
        stats['majority_class']['mask'] = np.argmax(stats['counts']['mask'])
        
        return new_info, stats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[torch.Tensor], dict]:
        """Gets sample from dataset:
        - Pads the obtained values to `win_max_length` seconds
        - Augments the obtained window
        - Extracts preliminary deep features if `processor` is set

        Args:
            index (int): Index of sample from info list

        Returns:
            tuple[torch.Tensor, dict[torch.Tensor], dict]: x, Y[gender, age], sample_info
        """
        data = self.info[index]
        a_data = load_data(os.path.join(self.full_features_path, 
                                        data['fn'].replace('.wav', '_{0}.dat'.format(data['w_idx']))))

        if self.transform:
            a_data = self.transform(a_data)

        if self.data_preprocessor:
            a_data = self.data_preprocessor(a_data)
        else:
            a_data = torch.nn.functional.pad(a_data, (0, max(0, int(self.win_max_length * self.sr) - len(a_data))), mode='constant')

        # OHE
        gen_value = [data['gen']]
        age_value = [data['age']]

        sample_info = {
            'filename': data['fn'],
            'start_t': data['start'] / self.sr,
            'end_t': data['end'] / self.sr,
            'start_f': data['start'],
            'end_f': data['end'],
            'corpus_name': self.corpus_name,
        }

        y = {'gen': torch.LongTensor(gen_value).squeeze(), 'age': torch.FloatTensor(age_value).squeeze()}
        return torch.FloatTensor(a_data), y, [sample_info]

    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of info list
        """
        return len(self.info)
