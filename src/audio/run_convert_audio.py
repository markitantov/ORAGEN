"""
This is the script for extracting audio from video with/without filtering speech.
"""

import os
import wave
import shutil
import subprocess
import pandas as pd

import sox
from tqdm import tqdm


from configs.unimodal_config import data_config


def convert_without_filtering(inp_path: str, 
                              out_path: str, 
                              checking: bool = True) -> None:
    """Convert video to audio using ffmpeg

    Args:
        inp_path (str): Input file path
        out_path (str): Output file path
        checking (bool, optional): Used for checking paths of the ffmpeg command. Defaults to True.
    """
    out_dirname = os.path.dirname(out_path)
    os.makedirs(out_dirname, exist_ok=True)

    # sample rate 16000
    command = f"ffmpeg -y -i {inp_path} -async 1 -vn -acodec pcm_s16le -ar 16000 -ac 1 {out_path}"
       
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        

def convert_video_to_audio(data_config: dict, 
                           db: str,
                           filtering: bool = False,
                           checking: bool = True) -> None:
    """Loops through the directory, and convert audio using ffmpeg.

    Args:
        data_config (dict): Dictonary info of database
        db (str): Database
        checking (bool, optional): Used for checking paths of the spleeter/ffmpeg commands. Defaults to True.
    """
    keyword = 'audio_file_path'
    data_df = pd.read_csv(data_config['LABELS_FILE_PATH'])

    for a_file in tqdm(sorted(list(set(data_df[keyword].values)))):
        if os.path.exists(os.path.join(data_config['RAW_DATA_ROOT'], a_file)):
            convert_without_filtering(os.path.join(data_config['RAW_DATA_ROOT'], a_file),
                                      os.path.join(data_config['PROCESSED_DATA_ROOT'], a_file.replace('.mp3', '.wav').replace('.m4a', '.wav')), 
                                      checking=checking)
        else:
            print('Error: ', os.path.join(data_config['RAW_DATA_ROOT'], a_file))


if __name__ == "__main__":    
    db_for_convert = ['COMMONVOICE', 'VOXCELEB2']
    for db in db_for_convert:
        print('Converting {}'.format(db))
        convert_video_to_audio(data_config=data_config[db], db=db, checking=False)