import sys

sys.path.append('src')

import os
import pprint
import pickle
import datetime
from copy import deepcopy

import pandas as pd

import torch
import torch.nn as nn

from torchvision import transforms
from transformers import AutoConfig

from configs.multimodal_bm_config import data_config as dconf
from configs.multimodal_bm_config import training_config as tconf

from fusion.data.agender_multimodal_features_dataset import AGenderMultimodalFeaturesDataset

from fusion.features.common import FeaturesType

from common.data.grouping import mask_grouping
from common.data.common import define_context_length, DatasetType

from fusion.models.multimodal_models import MaskAgenderAVModelV1, MaskAgenderAVModelV2, MaskAgenderAVModelV3, MaskAgenderAVModelV4

from fusion.augmentation.modality_augmentation import ModalityDropAugmentation

from common.loss.loss import MaskAgenderLoss

from common.utils.accuracy import *

from common.net_trainer.multitask_net_trainer import MultiTaskNetTrainer

from common.utils.common import get_source_code, define_seed, AttrDict, wait_for_it
  

def main(d_config: dict, t_config: dict) -> None:
    """Trains with configuration in the following steps:
    - Defines datasets names
    - Defines data augmentations
    - Defines data preprocessor
    - Defines datasets
    - Defines dataloaders
    - Defines measures
    - Defines NetTrainer
    - Defines model
    - Defines weighted loss, optimizer, scheduler
    - Runs NetTrainer 

    Args:
        d_config (dict): Data configuration dictionary
        t_config (dict): Training configuration dictionary
    """
    # Defining class names
    corpora_names = list(d_config.keys())
    c_names = d_config[corpora_names[0]]['C_NAMES']

    logs_root = t_config['LOGS_ROOT']
        
    features_root = t_config['FEATURE_EXTRACTOR']['FEATURES_ROOT']
    features_file_name = t_config['FEATURE_EXTRACTOR']['FEATURES_FILE_NAME']
    win_max_length = t_config['FEATURE_EXTRACTOR']['WIN_MAX_LENGTH']
    win_shift = t_config['FEATURE_EXTRACTOR']['WIN_SHIFT']
    win_min_length = t_config['FEATURE_EXTRACTOR']['WIN_MIN_LENGTH']
    features_type = t_config['FEATURE_EXTRACTOR']['FEATURES_TYPE']
    sr = t_config['FEATURE_EXTRACTOR']['SR']
        
    model_cls = t_config['MODEL']['cls']
    model_args = t_config['MODEL']['args']
    
    num_epochs = t_config['NUM_EPOCHS']
    batch_size = t_config['BATCH_SIZE']
    augmentation = t_config['AUGMENTATION']
    
    source_code = 'Data configuration:\n{0}\nTraining configuration:\n{1}\n\nSource code:\n{2}'.format(
        pprint.pformat(d_config),
        pprint.pformat(t_config),
        get_source_code([main, model_cls, AGenderMultimodalFeaturesDataset, MultiTaskNetTrainer]))
    
    # Defining datasets 
    ds_names = {
        'train': 'train',
        'test': 'test',
    }
    
    c_names_to_display = {}
    for task, class_names in c_names.items():
        c_names_to_display[task] = [cn.capitalize() for cn in class_names]
    
    # Defining metadata
    metadata_info = {}
    for corpus_name in corpora_names:
        corpus_labels_metadata = pd.read_csv(d_config[corpus_name]['LABELS_FILE_PATH'])
        with open(d_config[corpus_name]['VAD_FILE_PATH'], 'rb') as handle:
            corpus_vad_metadata = pickle.load(handle)
        
        metadata_info[corpus_name] = {}
        for ds in ds_names:
            condition_list = corpus_labels_metadata['subset'].isin(['train', 'devel', 'dev']) if 'train' in ds else corpus_labels_metadata['subset'].isin(['test'])

            metadata_info[corpus_name][ds] = {
                'data_root': d_config[corpus_name]['PROCESSED_DATA_ROOT'],
                'labels_metadata': corpus_labels_metadata[condition_list],
                'features_file_name': '{0}_{1}_{2}'.format(corpus_name, ds_names[ds].upper(), features_file_name),
                'vad_metadata': corpus_vad_metadata
            }
    
    # Defining datasets
    datasets = {}
    datasets_stats = {}
    for corpus_name in corpora_names:
        datasets[corpus_name] = {}
        datasets_stats[corpus_name] = {}

        for ds in ds_names:
            datasets[corpus_name][ds] = AGenderMultimodalFeaturesDataset(data_root=metadata_info[corpus_name][ds]['data_root'],
                                                                         labels_metadata=metadata_info[corpus_name][ds]['labels_metadata'], 
                                                                         features_root=features_root,
                                                                         features_file_name=metadata_info[corpus_name][ds]['features_file_name'],
                                                                         corpus_name=corpus_name,
                                                                         gender_num_classes=len(c_names['gen']),
                                                                         features_type=features_type,
                                                                         channels=['c'],
                                                                         mask_types=[0, 1, 2, 3, 4, 5],
                                                                         vad_metadata=metadata_info[corpus_name][ds]['vad_metadata'],
                                                                         sr=sr, win_max_length=win_max_length, win_shift=win_shift, 
                                                                         win_min_length=win_min_length,
                                                                         include_mask=True,
                                                                         transform=ModalityDropAugmentation() if ('train' in ds) and (augmentation) else None)

            datasets_stats[corpus_name][ds] = datasets[corpus_name][ds].stats

    # Defining dataloaders
    dataloaders = {}
    for ds in ds_names:
        if 'train' in ds:
            dataloaders[ds] = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([datasets[corpus_name]['train'] for corpus_name in corpora_names]),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
        else:
            for corpus_name in corpora_names:
                dataloaders['{0}_{1}'.format(ds, corpus_name)] = torch.utils.data.DataLoader(
                    datasets[corpus_name][ds],
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True
                )
        
    # Defining measures, measure with 0 index is main measure
    measures = [
        MAEMeasure('age_MAE'), # main measure
        UARMeasure('gen_UAR'), 
        MacroF1Measure('gen_MacroF1'),
        PrecisionMeasure('gen_Precision'),
        UARMeasure('mask_UAR'),
    ]
    
    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    experiment_name = 'w-{4}MultiModalMaskAgender{3}-{0}-{1}-{2}'.format(len(c_names['gen']),
                                                          model_cls.__name__.replace('-', '_').replace('/', '_'),
                                                          datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
                                                          {1: 'Early', 2: 'Inter', 3: 'Late'}[int(features_type)], 
                                                          'Augmentation' if augmentation else '')
    
    # Defining NetTrainer 
    net_trainer = MultiTaskNetTrainer(log_root=logs_root,
                                      experiment_name=experiment_name,
                                      c_names=c_names,
                                      measures=measures,
                                      device=device,                  
                                      final_activations={'gen': nn.Softmax(dim=-1), 'age': nn.Sigmoid(), 'mask': nn.Softmax(dim=-1)},
                                      group_predicts_fn=mask_grouping,
                                      source_code=source_code,
                                      c_names_to_display=c_names_to_display)
    
    model = model_cls(features_type=features_type, checkpoint_path=model_args['checkpoint_path'])
    model.to(device)
    
    # Defining weighted loss
    gender_class_sample_count = [sum(x) for x in zip(*[datasets_stats[corpus_name]['train']['counts']['gen'] for corpus_name in corpora_names])]
    mask_class_sample_count = [sum(x) for x in zip(*[datasets_stats[corpus_name]['train']['counts']['mask'] for corpus_name in corpora_names])]
    loss = MaskAgenderLoss(
        mask_weights=torch.Tensor(mask_class_sample_count / sum(mask_class_sample_count)).to(device),
        gender_weights=torch.Tensor(gender_class_sample_count / sum(gender_class_sample_count)).to(device), 
        gender_alpha=1, 
        age_alpha=1,
        mask_alpha=1,
    )
    
    # Defining optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)

    # Defining scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10, T_mult=2)

    model, max_perf = net_trainer.run(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler,
                                      num_epochs=num_epochs, dataloaders=dataloaders, datasets_stats=datasets_stats, log_epochs=[i for i in range(0, 10)])

    for phase, perf in max_perf.items():
        if 'train' in phase:
            continue

        print()
        print(phase.capitalize())
        print('Epoch: {}, Max performance:'.format(max_perf[phase][str(measures[0])]['epoch']))
        print([metric for metric in max_perf[phase][str(measures[0])]['performance']])
        print([max_perf[phase][str(measures[0])]['performance'][metric] for metric in max_perf[phase][str(measures[0])]['performance']])
        print()
    

def run_training() -> None:
    """Wrapper for training 
    """
    d_config = dconf
    
    models = [
        # {'cls': MaskAgenderAVModelV1, 'checkpoint_path': 'w-AugmentationMultiModalFinalInter-2-AVModelV3-2024.10.18-08.34.16/models/epoch_1.pth'},
        # {'cls': MaskAgenderAVModelV2, 'checkpoint_path': 'w-AugmentationMultiModalFinalInter-2-AVModelV3-2024.10.18-08.34.16/models/epoch_1.pth'},
        # {'cls': MaskAgenderAVModelV3, 'checkpoint_path': 'w-AugmentationMultiModalFinalInter-2-AVModelV3-2024.10.18-08.34.16/models/epoch_1.pth'},
        {'cls': MaskAgenderAVModelV4, 'checkpoint_path': None},
    ]

    features_types = [FeaturesType.INTERMEDIATE]
    
    for features_type in features_types:
        for idx, model_info in enumerate(models):
            corpora_names = ['VOXCELEB2', 'BRAVEMASKS']
            # corpora_names = ['VOXCELEB2']
            # corpora_names = ['BRAVEMASKS']
            new_d_config = {corpus_name: d_config[corpus_name] for corpus_name in corpora_names}
            
            t_config = deepcopy(tconf)
            t_config['FEATURE_EXTRACTOR']['FEATURES_TYPE'] = features_type

            t_config['FEATURE_EXTRACTOR']['FEATURES_FILE_NAME'] = {1: 'EARLYF_SAMPLES', 2: 'INTERF_SAMPLES', 3: 'LATEF_SAMPLES'}[int(features_type)]
                
            t_config['MODEL']['cls'] = model_info['cls']
            t_config['MODEL']['args'] = {'checkpoint_path': None}
            # t_config['MODEL']['args'] = {'checkpoint_path': os.path.join('/media/maxim/WesternDigital/aGender2024/MultimodalNoMask', model_info['checkpoint_path'])}
            t_config['AUGMENTATION'] = True
            t_config['NUM_EPOCHS'] = 100
            
            main(d_config=new_d_config, t_config=t_config)


if __name__ == "__main__":
    run_training()