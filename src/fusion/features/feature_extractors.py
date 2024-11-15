import sys

sys.path.append('src')

import pickle

import torch
import torch.nn as nn

from transformers import AutoConfig

from common.data.common import read_img
from audio.models.audio_models import AGenderAudioW2V2Model
from video.models.video_models import VIT_DPAL
from common.data.data_preprocessors import Wav2Vec2DataPreprocessor, ViTDataPreprocessor
from fusion.features.common import FeaturesType
    
    
class BaseFeatureExtractor:
    def __init__(self) -> None:
        """Base Feature Extractor class
        """
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Base Feature Extractor implementation

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Features
        """
        return x
    
    
class AudioFeatureExtractor(BaseFeatureExtractor): 
    def __init__(self, features_type: FeaturesType, sr: int = 16000, win_max_length: int = 4) -> None:
        """Audio Feature Extractor

        Args:
            features_type (FeaturesType): Type of features.
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
        """
        self.sr = sr
        self.win_max_length = win_max_length
        self.features_type = features_type
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.output_size = 3
        
        self.processor = Wav2Vec2DataPreprocessor(model_name)
        
        self.model = AGenderAudioW2V2Model.from_pretrained(pretrained_model_name_or_path=model_name, 
                                           config=model_config)
        
        checkpoint = torch.load('/media/maxim/WesternDigital/aGender2024/w-2-AGender_Audio_W2V2Model-2024.09.07-18.40.39/models/epoch_1.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts acoustic features
        Apply padding to max length of audio

        Args:
            waveform (torch.Tensor): Input waveform

        Returns:
            torch.Tensor: Extracted features
        """
        waveform = self.processor(waveform).unsqueeze(0).to(self.device)

        with torch.no_grad():        
            if self.features_type == FeaturesType.EARLY:
                features = self.model.early_features(waveform)
            elif self.features_type == FeaturesType.INTERMEDIATE:
                features = self.model.intermediate_features(waveform)
            else:
                features = self.model.late_features(waveform)
            
        return features.detach().cpu().squeeze()
    
    
class VideoFeatureExtractor(BaseFeatureExtractor): 
    def __init__(self, features_type: FeaturesType, win_max_length: int = 4) -> None:
        """Video Feature Extractor

        Args:
            features_type (FeaturesType): Type of features.
            win_max_length (int, optional): Max length of window. Defaults to 4.
        """
        self.win_max_length = win_max_length
        self.features_type = features_type
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                
        self.processor = ViTDataPreprocessor()
        
        self.model = VIT_DPAL().to(self.device)
        checkpoint = torch.load('/media/maxim/Programs/Projects/AGender/models/combined_vit_DPAL_weights.pth')
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def __call__(self, img_paths: list[str]) -> torch.Tensor:
        """Extracts visual features
        Apply duplicating to max length

        Args:
            img_paths (list[str]): Path of images

        Returns:
            torch.Tensor: Extracted features
        """
        image_paths = img_paths[:self.win_max_length] + [img_paths[-1]]*(self.win_max_length - len(img_paths))
        images = [read_img(img_path) for img_path in image_paths]
        prepared_images = [self.processor(img) for img in images]
        batched_images = torch.stack(prepared_images, dim=0).to(self.device)

        with torch.no_grad():        
            if self.features_type == FeaturesType.EARLY:
                features = self.model.early_features(batched_images)
            elif self.features_type == FeaturesType.INTERMEDIATE:
                features = self.model.intermediate_features(batched_images)
            else:
                features = self.model.late_features(batched_images)
                                            
        return features.detach().cpu()
