import numpy as np
from transformers import Wav2Vec2Processor, ViTImageProcessor
import torch


class BaseDataPreprocessor:
    def __init__(self) -> None:
        """Base Data Preprocessor class
        """
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Base Data Preprocessor implementation

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Preprocessed data
        """
        return x


class Wav2Vec2DataPreprocessor(BaseDataPreprocessor): 
    def __init__(self, preprocessor_name: str = 'facebook/wav2vec2-large-robust', 
                 sr: int = 16000, win_max_length: int = 4, return_attention_mask: bool = False) -> None:
        """Wav2Vec Data Preprocessor

        Args:
            preprocessor_name (str, optional): Preprocessor name in transformers library. 
                                               Defaults to 'facebook/wav2vec2-large-robust'.
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
            return_attention_mask: (bool, optional): Return attention mask or not. Defaults to False
        """
        self.sr = sr
        self.win_max_length = win_max_length
        
        self.return_attention_mask = return_attention_mask
        self.processor = Wav2Vec2Processor.from_pretrained(preprocessor_name)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features for wav2vec using 'facebook/wav2vec2-large-robust' preprocessor 
        from transformers library
        Apply padding to max length of audio

        Args:
            x (torch.Tensor): Input data

        Returns:
            np.ndarray: Preprocessed data
        """
        a_data = self.processor(x, sampling_rate=self.sr, return_tensors="pt", 
                                padding='max_length', max_length=self.sr * self.win_max_length)
        return a_data if self.return_attention_mask else a_data["input_values"][0]


class HuBERTDataPreprocessor(BaseDataPreprocessor): 
    def __init__(self, preprocessor_name: str = 'facebook/wav2vec2-large-robust', 
                 sr: int = 16000, win_max_length: int = 4, return_attention_mask: bool = False) -> None:
        """Wav2Vec Data Preprocessor

        Args:
            preprocessor_name (str, optional): Preprocessor name in transformers library. 
                                               Defaults to 'facebook/wav2vec2-large-robust'.
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
            return_attention_mask: (bool, optional): Return attention mask or not. Defaults to False
        """
        self.sr = sr
        self.win_max_length = win_max_length
        
        self.return_attention_mask = return_attention_mask
        self.processor = Wav2Vec2Processor.from_pretrained(preprocessor_name)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features for wav2vec using 'facebook/wav2vec2-large-robust' preprocessor 
        from transformers library
        Apply padding to max length of audio

        Args:
            x (torch.Tensor): Input data

        Returns:
            np.ndarray: Preprocessed data
        """
        a_data = self.processor(x, sampling_rate=self.sr, return_tensors="pt", 
                                padding='max_length', max_length=self.sr * self.win_max_length)
        return a_data if self.return_attention_mask else a_data["input_values"][0]


class ViTDataPreprocessor(BaseDataPreprocessor): 
    def __init__(self, preprocessor_name: str = 'nateraw/vit-age-classifier') -> None:
        """ViT Data Preprocessor

        Args:
            preprocessor_name (str, optional): Preprocessor name in transformers library. 
                                               Defaults to 'nateraw/vit-age-classifier'.
        """
        self.processor = ViTImageProcessor.from_pretrained(preprocessor_name)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features for wav2vec using 'nateraw/vit-age-classifier' preprocessor 
        from transformers library

        Args:
            x (torch.Tensor): Input data

        Returns:
            np.ndarray: Preprocessed data
        """
        a_data = self.processor(x, return_tensors="pt")['pixel_values']
        return a_data[0]

    
if __name__ == "__main__":
    sampling_rate = 16000
    device = torch.device('cpu')
    inp_v = torch.zeros((sampling_rate * 4)).to(device)
    
    data_preprocessor = Wav2Vec2DataPreprocessor('facebook/hubert-large-ls960-ft', sr=16000, return_attention_mask=False)
    y = data_preprocessor(inp_v)
    print(y)