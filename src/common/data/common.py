import sys

sys.path.append('src')

import os
import re
import pickle
from enum import IntEnum

import numpy as np

import PIL

import torch
import torchaudio


def read_audio(filepath: str, sample_rate: int) -> torch.Tensor:
    """Read audio using torchaudio

    Args:
        filepath (str): Path to audio file
        sample_rate (int): Sample rate of audio file

    Returns:
        torch.Tensor: Wave
    """
    full_wave, sr = torchaudio.load(filepath)

    if full_wave.size(0) > 1:
        full_wave = full_wave.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        full_wave = transform(full_wave)

    full_wave = full_wave.squeeze(0)
    
    return full_wave


def save_data(data: any, filename: str) -> None:
    """Dumps data to pickle

    Args:
        data (any): Data
        filename (str): Filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename: str) -> any:
    """Reads data from pickle

    Args:
        filename (str): Filename
    
    Returns:
        data (any): Data
    """
    data = None
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)

    return data


def find_intersections(x: list[dict], y: list[dict], min_length: float = 0) -> list[dict]:
    """Find intersections of two lists of dicts with intervals

    Args:
        x (list[dict]): First list
        y (list[dict]): Second list
        min_length (float, optional): Minimum length of intersection. Defaults to 0.

    Returns:
        list[dict]: Windows with VAD intersection
    """

    timings = []
    # `i` is pointer for `x`, `j` - for `y`
    i = 0
    j = 0

    while i < len(x) and j < len(y):
        # Left bound for intersecting segment
        l = max(x[i]['start'], y[j]['start'])
         
        # Right bound for intersecting segment
        r = min(x[i]['end'], y[j]['end'])

        if l <= r: # If segment is valid 
            if r - l >= min_length: # if length of intersection not less then `min_length` seconds
                timings.append({'start': l, 'end': r})
         
        # If i-th interval's right bound is 
        # smaller increment i else increment j
        if x[i]['end'] < y[j]['end']:
            i += 1
        else:
            j += 1

    return timings


def slice_audio(start_time: float, end_time: float, win_max_length: float, win_shift: float, win_min_length: float) -> list[dict]:
    """Slices audio on windows

    Args:
        start_time (float): Start time of audio
        end_time (float): End time of audio
        win_max_length (float): Window max length
        win_shift (float): Window shift
        win_min_length (float): Window min length

    Returns:
        list[dict]: List of dict with timings, f.e.: {'start': 0, 'end': 12}
    """    

    if end_time < start_time:
        return []
    elif (end_time - start_time) > win_max_length:
        timings = []
        while start_time < end_time:
            end_time_chunk = start_time + win_max_length
            if end_time_chunk < end_time:
                timings.append({'start': start_time, 'end': end_time_chunk})
            elif end_time_chunk == end_time: # if tail exact `win_max_length` seconds
                timings.append({'start': start_time, 'end': end_time_chunk})
                break
            else: # if tail less then `win_max_length` seconds
                if end_time - start_time < win_min_length: # if tail less then `win_min_length` seconds
                    break
                
                timings.append({'start': start_time, 'end': end_time})
                break

            start_time += win_shift
        return timings
    else:
        return [{'start': start_time, 'end': end_time}]
    
    
def generate_features_file_name(**kwargs: dict) -> str:
    """Generates features file name

    Returns:
        str: features file name
    """
    return '{0}{1}{2}{3}'.format('VAD' if kwargs['vad_metadata'] else '',
                                    kwargs['win_max_length'], kwargs['win_shift'], kwargs['win_min_length'])
    
    
def define_context_length(win_max_length: int = 4) -> int:
    """Define context length in models

    Args:
        win_max_length (int): Max length of window. Defaults to 4.

    Returns:
        int: Context length
    """
    return {
        1: 49,
        2: 99,
        3: 149,
        4: 199
    }[win_max_length]


def gender_label_to_int(value: str, num_classes: int) -> int:
    """Convert gender value to label
    Child -> 0 class
    Female -> 1 class
    Male -> 2 class

    Args:
        value (str): Gender label
        num_classes (str): Number of classes: binary or ternary problem

    Returns:
        int: Converted Gender label
    """
    return {
        2: {
            'female': 0,
            'male': 1,
        },
        3: {
            'child': 0,
            'female': 1,
            'male': 2
        }
    }[num_classes][value]


def mask_label_to_int(value: str) -> int:
    """Convert mask value to label
    No mask -> 0 class
    Tissue mask -> 1 class
    Medical mask -> 2 class
    Protective mask (ffp2/ffp3) -> 3 class
    Respirator -> 4 class
    Protective face shield -> 5 class

    Args:
        value (str): Mask label value

    Returns:
        int: Converted Mask label
    """
    return {
        'No mask': 0,
        'Tissue mask': 1,
        'Medical mask': 2,
        'Protective mask (ffp2/ffp3)': 3,
        'Respirator': 4,
        'Protective face shield': 5,
    }[value]
    
    
def read_img(path: str):
    """Read image and convert to numpy

    Args:
        path (str): Path of image

    Returns:
        PIL image
    """
    img = PIL.Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


class DatasetType(IntEnum):
    """
    Dataset type Enum
    """
    AUDIO: int = 1
    VIDEO: int = 2
    BOTH: int = 3


if __name__ == "__main__":
    print(slice_audio(12, 6.1, 4, 2, 2))