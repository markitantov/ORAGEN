import sys

sys.path.append('src')

from common.data.data_preprocessors import BaseDataPreprocessor


data_config: dict = {
    'AGENDER': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
        },
    },
    'TIMIT': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
        },
    },
    'COMMONVOICE': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
        },
    },
    'VOXCELEB2': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
            'mask': ['No mask', 'Tissue mask', 'Medical mask', 
                     'Protective mask (ffp2/ffp3)', 'Respirator', 'Protective face shield'],
        },
    },
    'BRAVEMASKS': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
            'mask': ['No mask', 'Tissue mask', 'Medical mask', 'Protective mask (ffp2/ffp3)', 'Respirator', 'Protective face shield'],
        },
    },
}

training_config: dict = {
    'LOGS_ROOT': '',
    'MODEL': {
        'cls': None,
        'args': { 
        }
    },
    'FEATURE_EXTRACTOR': {
        'FEATURES_ROOT': '',
        'FEATURES_FILE_NAME': 'SAMPLES',
        'WIN_MAX_LENGTH': 4,
        'WIN_SHIFT': 2,
        'WIN_MIN_LENGTH': 1,
        'SR': 16000,
    },
    'NUM_EPOCHS': 10,
    'BATCH_SIZE': 128,
    'AUGMENTATION': True
}