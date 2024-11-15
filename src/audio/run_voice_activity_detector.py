"""
This is the script for extracting audio from video with/without filtering speech.
"""

import os
import pickle
import pandas as pd

import torch
from tqdm import tqdm

from configs.unimodal_bm_config import data_config


model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps, _, read_audio, _, _) = utils


def detect_speech(inp_path: str,
                  sampling_rate: int = 16000) -> list[dict]:
    """Finds speech segments using VAD

    Args:
        inp_path (str): Input file path
        sampling_rate (int, optional): Sampling rate of audio. Defaults to 16000.

    Returns:
        list[dict]: list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """
    
    wav = read_audio(inp_path, sampling_rate=sampling_rate)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    
    return speech_timestamps


def run_vad(data_config: dict,
            db: str) -> None:
    """Loops through the directory, and run VAD on DB.

    Args:
        data_config (dict): Dictonary info of database
        db (str): Database
    """
    keyword = 'audio_file_path'
    data_df = pd.read_csv(data_config['LABELS_FILE_PATH'])
    data_df[keyword] = data_df[keyword].apply(lambda x: x.replace('.mp3', '.wav').replace('.m4a', '.wav'))
    
    res = {}
    for a_file in tqdm(sorted(list(set(data_df[keyword].values)))):
        if os.path.exists(os.path.join(data_config['PROCESSED_DATA_ROOT'], a_file)):
            speech_timestamps = detect_speech(inp_path=os.path.join(data_config['PROCESSED_DATA_ROOT'], a_file))
            res[a_file] = speech_timestamps
        else:
            print('Error: ', os.path.join(data_config['PROCESSED_DATA_ROOT'], a_file))

    with open(data_config['VAD_FILE_PATH'], 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    dbs = list(data_config.keys())
    
    for db in dbs:
        if db in ['COMMONVOICE', 'AGENDER', 'TIMIT', 'VOXCELEB2']:
            continue

        print('Starting VAD on {}'.format(db))
        run_vad(data_config=data_config[db], db=db)
