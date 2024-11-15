import numpy as np
from sklearn import metrics
from enum import Enum


class MeasureMode(Enum):
    """
    Measure type Enum
    Defines which value of measure "better"
    """
    LESS: int = 1
    MORE: int = 2


class BaseMeasure():
    """Base Measure class
    Args:
        name (str, optional): Performance measure name
        mode (MeasureMode, optional): Defines which value of measure "better". Defaults to MeasureMode.MORE.
        protection (str, optional): Allows you to calculate performance measure only for a specific corpus. Defaults to None.
    """
    def __init__(self, name: str = 'Measure', mode=MeasureMode.MORE, protection: str = None) -> None:
        self.name = name
        self.mode = mode
        self.protection = protection
    
    def __call__(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        pass
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name


class MAEMeasure(BaseMeasure): 
    """MAE measure
    """
    def __init__(self, name: str = 'MAE',  mode=MeasureMode.LESS, protection: str = None) -> None:
        super().__init__(name, mode, protection)
    
    def __call__(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates Mean Absolute Error

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: MAE value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)
        res = metrics.mean_absolute_error(targets, predicts)
        return res * 100


class UARMeasure(BaseMeasure): 
    """UAR measure
    """
    def __init__(self, name: str = 'UAR', mode=MeasureMode.MORE, protection: str = None) -> None:
        super().__init__(name, mode, protection)
    
    def __call__(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates Unweighted Average Recall

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: UAR value
        """
        targets = np.array(targets).astype(int)
        predicts = np.array(predicts).astype(int)
        res = metrics.recall_score(targets, predicts, average='macro')
        return res * 100


class MacroF1Measure(BaseMeasure): 
    """MacroF1 measure
    """
    def __init__(self, name: str = 'MacroF1', mode=MeasureMode.MORE, protection: str = None) -> None:
        super().__init__(name, mode, protection)
    
    def __call__(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates MacroF1 score

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: MacroF1 value
        """
        targets = np.array(targets).astype(int)
        predicts = np.array(predicts).astype(int)
        res = metrics.f1_score(targets, predicts, average='macro')
        return res * 100


class PrecisionMeasure(BaseMeasure): 
    """Precision measure
    """
    def __init__(self, name: str = 'Precision', mode=MeasureMode.MORE, protection: str = None) -> None:
        super().__init__(name, mode, protection)
    
    def __call__(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates Precision

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: Precision value
        """
        targets = np.array(targets).astype(int)
        predicts = np.array(predicts).astype(int)
        res = metrics.precision_score(targets, predicts, average='macro')
        return res * 100
    

if __name__ == "__main__":
    measures = [MAEMeasure, UARMeasure, MacroF1Measure, PrecisionMeasure]
    for measure in measures:
        m = measure()
        print(m)
        print('{0} = {1:.3f}%'.format(m, m([1, 2, 3], [3, 2, 3])))