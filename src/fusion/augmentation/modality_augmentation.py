import torch

from common.data.common import load_data
from fusion.data.agender_multimodal_dataset import DatasetType

class ModalityDropAugmentation(torch.nn.Module):
    """Randomly dropes one of the modality zeroing out all elements
    Generates value (uniform distribution) on specified limits
    
    Logic:
    if generated_value in limits[0]: drop audio modality
    if generated_value in limits[1]: do nothing
    if generated_value in limits[2]: drop video modality
    
    Args:
        limits (list[tuple[int, int]], optional): Limits of generated value. Defaults to [(0, 20), (20, 80), (80, 100)].
    """
    def __init__(self, limits: list[tuple[int, int]] = None) -> None:
        super(ModalityDropAugmentation, self).__init__()
        self.limits = limits if limits else [(0, 25), (25, 75), (75, 100)]
        self.min_l = self.limits[0][0]
        self.max_l = self.limits[2][1]
    
    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Generates value (uniform distribution) on specified limits
        and drop (zeroing out) modalities

        Args:
            x (list[torch.Tensor]): Input (audio, video) tensor

        Returns:
            list[torch.Tensor]: Modified (audio, video) tensor
        """
        a, v = x
        # generate uniformly distributed value on [min_l, max_l].
        choise = torch.FloatTensor(1).uniform_(self.min_l, self.max_l)
        if self.limits[0][0] <= choise < self.limits[0][1]:
            a = torch.zeros(a.shape)
        elif self.limits[1][0] <= choise < self.limits[1][1]:
            return a, v
        elif self.limits[2][0] <= choise <= self.limits[2][1]:
            v = torch.zeros(v.shape)
        
        return a, v


class MultiAugment(torch.nn.Module):
    def __init__(self, features_type, dataset_type) -> None:
        super(MultiAugment, self).__init__()
        self.dataset_type = dataset_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize channels of an input image

        Args:
            x (torch.Tensor): Input tensor of shape (224, 224, 3)

        Returns:
            torch.Tensor: Normalized tensor (3, 224, 224)
        """
        a_data, v_data = x
        if self.dataset_type == DatasetType.BOTH:
            return a_data, v_data
        
        if self.dataset_type == DatasetType.AUDIO:
            v_data = torch.zeros(v_data.shape)
        else:
            a_data = torch.zeros(a_data.shape)
        
        return a_data, v_data