import sys

sys.path.append('src')

import os
import re
import pickle


import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torchvision

from torch.utils.data import Dataset

from configs.unimodal_config import data_config as conf

from common.data.common import load_data, save_data, slice_audio, find_intersections, read_audio, generate_features_file_name, gender_label_to_int, mask_label_to_int, read_img, DatasetType
from fusion.features.feature_extractors import AudioFeatureExtractor, VideoFeatureExtractor, FeaturesType


class AGenderMultimodalFeaturesDataset(Dataset):
    def __init__(self, data_root: str, labels_metadata: pd.DataFrame, 
                 features_root: str, features_file_name: str, 
                 corpus_name: str, gender_num_classes: int,
                 features_type: FeaturesType,
                 channels: list[str] = ['c'],
                 mask_types: list[int] = [0, 1, 2, 3, 4, 5],
                 vad_metadata: dict[list] = None,
                 sr: int = 16000, win_max_length: int = 4, win_shift: int = 2, win_min_length: int = 0, 
                 include_mask: bool = False,
                 dataset_type: DatasetType = DatasetType.BOTH,
                 transform: torchvision.transforms.transforms.Compose = None) -> None:
        """AGender multimodal dataset
        Preprocesses labels and data

        Args:
            data_root (str): Audio root dir
            labels_metadata (pd.DataFrame): Pandas labels
            features_root (str): File path for fast load features
            features_file_name (str): File name for fast load features
            corpus_name (str): Name of corpus
            gender_num_classes (int): Number of gender classes
            features_type (FeaturesType): Type of features
            channels (list[str]): Filter for channels. Can be ['c', 'r', 'l'] for using all filters. Defaults to ['c'].
            mask_types (list[int]): Filter for mask type according to `mask_label_to_int`. Defaults to [0, 1, 2, 3, 4, 5].
            vad_metadata (dict[list], optional): VAD information. Defaults to None.
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
            win_shift (int, optional): Shift length of window. Defaults to 2.
            win_min_length (int, optional): Min length of window. Defaults to 0.
            include_mask (bool, optional): Include mask or not. Defaults to False.
            dataset_type (DatasetType, optional): Dataset type. Unimodal or multimodal. Defaults to DatasetType.AUDIO.
            transform (torchvision.transforms.transforms.Compose, optional): Augmentation methods (both). Defaults to None.
        """
        self.data_root = data_root
        self.labels_metadata = labels_metadata
        self.vad_metadata = vad_metadata
        
        self.sr = sr
        self.win_max_length = win_max_length
        self.win_shift = win_shift
        self.win_min_length = win_min_length
        self.include_mask = include_mask
        
        self.dataset_type = dataset_type
        self.channels = channels
        self.mask_types = mask_types
        
        self.av_transform = transform
        self.corpus_name = corpus_name
        self.gender_num_classes = gender_num_classes
        self.features_type = features_type

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
        - Drop audio duplicates
        - Reads audio
        - Slices audio
        - Finds windows intersections according to VAD
        - Saves features
        - Calculates label statistics
        """
        self.info = []
        
        audio_feature_extractor = AudioFeatureExtractor(features_type=self.features_type)
        video_feature_extractor = VideoFeatureExtractor(features_type=self.features_type)

        self.labels_metadata = self.labels_metadata.sort_values('audio_file_path').drop_duplicates(['audio_file_path'], keep='last').copy()
        for sample in tqdm(self.labels_metadata.to_dict('records')):
            sample_filename = sample['audio_file_path'].replace('.mp3', '.wav').replace('.m4a', '.wav')
            sample_fp = os.path.join(self.data_root, sample_filename)
            
            image_info = re.split(r'__s\d{3}', sample['image_file_path'])
                      
            sample_gen = sample['gender']
            sample_age = sample['age']
            sample_mask = 'No mask' if 'mask_type' not in sample else sample['mask_type']
            
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

            if ('lQxVumsa0QE/00053' in sample_filename) or ('x9K8-IfuOMg/00490' in sample_filename) \
            or ('x9K8-IfuOMg/00491' in sample_filename) or ('StiTPpXXhe0/00079' in sample_filename):
                continue

            for w_idx, window in enumerate(intersections):
                wave = full_wave[window['start']: window['end']].clone()
                images_fn = ['{0}__s{1}{2}'.format(image_info[0], 
                                                   str(i).zfill(3), 
                                                   image_info[1]) for i in range(int(window['start'] / self.sr) + 1, 
                                                                                 int(window['end'] / self.sr) + 1)]
                
                images_fp = [os.path.join(self.data_root, img_fn) for img_fn in images_fn]
                for idx, image in enumerate(images_fp):
                    if not os.path.exists(image):
                        import torchaudio
                        torchaudio.save('{1}_{0}.wav'.format(os.path.basename(sample_filename).split('.')[0], idx), wave.unsqueeze(0), self.sr)
                        print(image, int(window['start'] / self.sr) + 1, int(window['end'] / self.sr) + 1)                    
                
                features = {'acoustic_features': audio_feature_extractor(wave), 'visual_features': video_feature_extractor(images_fp)}
                
                self.info.append({
                    'fp': sample_fp,
                    'fn': sample_filename,
                    'img_fp': images_fp,
                    'img_fn': images_fn,
                    'w_idx': w_idx,
                    'start': window['start'],
                    'end': window['end'],
                    'gen': sample_gen,
                    'age': sample_age,
                    'mask': sample_mask,
                    'channel': sample['channel'] if 'channel' in sample else 'c',
                })
                
                save_data(features, os.path.join(self.full_features_path, 
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
            sample_channel =sample_info['channel'] if 'channel' in sample_info else 'c'

            if sample_channel not in self.channels: # filter channel
                continue

            if sample_mask not in self.mask_types: # filter mask_type
                continue
            
            new_info.append({
                'fp': sample_info['fp'],
                'fn': sample_info['fn'],
                'img_fp': sample_info['img_fp'],
                'img_fn': sample_info['img_fn'],
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
        
        # audio data
        av_data = load_data(os.path.join(self.full_features_path, 
                                         data['fn'].replace('.wav', '_{0}.dat'.format(data['w_idx']))))
            
        a_data = av_data['acoustic_features']
        v_data = av_data['visual_features']
          
        if self.dataset_type == DatasetType.VIDEO:
            a_data = torch.zeros(a_data.shape)
        elif self.dataset_type == DatasetType.AUDIO:
            v_data = torch.zeros(v_data.shape)
            
        if (self.dataset_type == DatasetType.BOTH) and (self.av_transform):
            a_data, v_data = self.av_transform((a_data, v_data))
        
        # OHE
        gen_value = [data['gen']]
        age_value = [data['age']]
        mask_value = [data['mask']]

        sample_info = {
            'filename': data['fn'],
            'start_t': data['start'] / self.sr,
            'end_t': data['end'] / self.sr,
            'start_f': data['start'],
            'end_f': data['end'],
            'corpus_name': self.corpus_name,
        }

        if self.include_mask:
            y = {
                'gen': torch.LongTensor(gen_value).squeeze(), 
                'age': torch.FloatTensor(age_value).squeeze(),
                'mask': torch.LongTensor(mask_value).squeeze(),
            }
        else:
            y = {
                'gen': torch.LongTensor(gen_value).squeeze(), 
                'age': torch.FloatTensor(age_value).squeeze()
            }
            
        return [torch.FloatTensor(a_data), torch.FloatTensor(v_data)], y, [sample_info]

    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of info list
        """
        return len(self.info)
