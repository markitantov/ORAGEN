import torch
import torch.nn.functional as F
import torch.nn as nn


class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Measures the agreement between two variables
    
    It is a product of
    - precision (pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)
    
    Interpretation
    - rho =  1: perfect agreement
    - rho =  0: no agreement
    - rho = -1: perfect disagreement
    
    Args:
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.
    """
    
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes CCC loss

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: 1 - CCC loss value
        """
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x, correction=0)
        y_s = torch.std(y, correction=0)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc


class AGenderLoss(nn.Module):
    def __init__(self, 
                 gender_weights: torch.Tensor = None,
                 gender_alpha: float = 1, age_alpha: float = 1) -> None:
        """Multitask loss function
    
        Args:
            gender_weights (torch.Tensor): Weights for gender. Defaults to None.
            gender_alpha (float, optional): Weighted coefficient for gender. Defaults to 1.
            age_alpha (float, optional): Weighted coefficient for age. Defaults to 1.
        """
        super(AGenderLoss, self).__init__()
        self.gender_alpha = gender_alpha
        self.age_alpha = age_alpha
        
        self.gender_loss = torch.nn.CrossEntropyLoss(weight=gender_weights)
        self.age_loss = CCCLoss()
        
        self.loss_values = {
            'gen': 0,
            'age': 0,
            'mask': 0,
        }

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes agender loss, which is sum of CrossEntropyLoss and CCC

        Args:
            predicts (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: loss value
        """
        # self.gender_loss_value = self.gender_loss(predicts['gen'], targets['gen'])
        # self.age_loss_value = self.age_loss(F.sigmoid(predicts['age']), targets['age']) # get predicts
        # return self.gender_alpha * self.gender_loss_value + self.age_alpha * self.age_loss_value
    
        self.loss_values['gen'] = self.gender_loss(predicts['gen'], targets['gen'])
        self.loss_values['age'] = self.age_loss(F.sigmoid(predicts['age']), targets['age']) # get predicts
        
        return self.gender_alpha * self.loss_values['gen'] + self.age_alpha * self.loss_values['age']


class MaskLoss(nn.Module):
    def __init__(self, 
                 mask_weights: torch.Tensor = None) -> None:
        """Mask loss function
    
        Args:
            mask_weights (torch.Tensor): Weights for mask classes. Defaults to None.
        """
        super(MaskLoss, self).__init__()        
        self.mask_loss = torch.nn.CrossEntropyLoss(weight=mask_weights)

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes mask type loss

        Args:
            predicts (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: loss value
        """
        return self.mask_loss(predicts, targets)
    

class MaskAgenderLoss(nn.Module):
    def __init__(self, 
                 mask_weights: torch.Tensor = None,
                 gender_weights: torch.Tensor = None,
                 gender_alpha: float = 1, age_alpha: float = 1, mask_alpha: float = 1) -> None:
        """Multitask loss function
    
        Args:
            gender_weights (torch.Tensor): Weights for gender. Defaults to None.
            gender_alpha (float, optional): Weighted coefficient for gender. Defaults to 1.
            age_alpha (float, optional): Weighted coefficient for age. Defaults to 1.
            mask_weights (torch.Tensor): Weights for mask classes. Defaults to None.
            mask_alpha (float, optional): Weighted coefficient for gender. Defaults to 1.
        """
        super(MaskAgenderLoss, self).__init__()
        
        self.gender_alpha = gender_alpha
        self.age_alpha = age_alpha
        self.mask_alpha = mask_alpha
        
        self.gender_loss = torch.nn.CrossEntropyLoss(weight=gender_weights)            
        self.age_loss = CCCLoss()
        self.mask_loss = torch.nn.CrossEntropyLoss(weight=mask_weights)
        
        self.loss_values = {
            'gen': 0,
            'age': 0,
            'mask': 0,
        }

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes sum of mask and agender loss

        Args:
            predicts (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: loss value
        """
        self.loss_values['gen'] = self.gender_loss(predicts['gen'], targets['gen'])
        self.loss_values['age'] = self.age_loss(F.sigmoid(predicts['age']), targets['age']) # get predicts
        self.loss_values['mask'] = self.mask_loss(predicts['mask'], targets['mask'])
        
        return self.gender_alpha * self.loss_values['gen'] + self.age_alpha * self.loss_values['age'] + self.mask_alpha * self.loss_values['mask']
