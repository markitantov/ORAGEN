import torch

class PreprocessInput(torch.nn.Module):
    """Preprocess augmentation according to default preprocessing of tensorflow model
    Adapted code from tensorflow code to pytorch
    """
    def __init__(self) -> None:
        super(PreprocessInput, self).__init__()
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize channels of an input image

        Args:
            x (torch.Tensor): Input tensor of shape (224, 224, 3)

        Returns:
            torch.Tensor: Normalized tensor (3, 224, 224)
        """
        x = x.to(torch.float32)
        
        # reverse order of 0 axe
        x = torch.flip(x, dims=(0,))
        x[0, :, :] -= 91.4953
        x[1, :, :] -= 103.8827
        x[2, :, :] -= 131.0912
        return x
    