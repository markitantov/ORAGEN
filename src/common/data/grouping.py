import sys

sys.path.append('src')

import numpy as np
import pandas as pd


def unimodal_grouping(targets: dict[list], predicts: dict[list], samples_info: dict[tuple], current_phase: str = None, datasets_stats: dict[dict] = None) -> dict[tuple]:
    """Unimodal grouping
    Args:
        targets (dict[list]): Dict of lists with targets for 'gen' and 'age'
        predicts (dict[list]): Dict of lists with predicts for 'gen' and 'age'
        samples_info (dict[tuple]): List of dicts with sample info
        current_phase (str): Current phase, can be train/dev/test. Defaults to None.
        datasets_stats: Metadata statistics for all corpora. Defaults to None.

    Returns:
        dict[tuple]: For each db tuple: 
                        targets - dict with keys ['gen', 'age']
                        predicts - dict with keys ['gen', 'age']
                        list(filenames)
    """
    res = {}
    filenames = []
    corpora_names = []
    for s_info in samples_info:
        filenames.extend(s_info['filename'])
        corpora_names.extend(s_info['corpus_name'])
    
    d_gen = {}
    d_gen['predicts'] = predicts['gen']
    d_gen['targets'] = targets['gen']
    d_gen['filenames'] = filenames
    d_gen['corpora_names'] = corpora_names
    df_all_gen = pd.DataFrame(d_gen)

    d_age = {}
    d_age['predicts'] = predicts['age']
    d_age['targets'] = targets['age']
    d_age['filenames'] = filenames
    d_age['corpora_names'] = corpora_names
    df_all_age = pd.DataFrame(d_age)

    ordered_corpora_names = sorted(list(set(corpora_names)))
    for corpus_name in ordered_corpora_names:        
        df_gen = df_all_gen[df_all_gen['corpora_names'] == corpus_name].copy()
        df_gen['predicts'] = df_gen['predicts'].apply(lambda x: np.argmax(x))
        df_gen = df_gen.groupby('filenames', as_index=False).agg(lambda x: pd.Series.mode(x)[0])
        df_gen = df_gen.sort_values('filenames')
        gen_targets = df_gen['targets'].to_list()
        gen_predicts = df_gen['predicts'].to_list()

        df_age = df_all_age[df_all_age['corpora_names'] == corpus_name].copy()
        df_age = df_age.drop('corpora_names', axis=1)
        df_age = df_age.groupby('filenames', as_index=False).mean()
        df_age = df_age.sort_values('filenames')
        
        age_targets = [round(elem, 2) for elem in df_age['targets'].to_list()]
        age_predicts = [round(elem, 2) for elem in df_age['predicts'].to_list()]

        filenames = df_gen['filenames'].to_list()
    
        if datasets_stats:
            current_phase = current_phase.split('_')[0]
            missing_files = set(datasets_stats[corpus_name][current_phase]['fns'].keys()) - set(filenames)
            for missing_file in missing_files:
                gen_targets.append(datasets_stats[corpus_name][current_phase]['fns'][missing_file]['gen'])
                gen_predicts.append(datasets_stats[corpus_name]['train']['majority_class']['gen'])

                age_targets.append(datasets_stats[corpus_name][current_phase]['fns'][missing_file]['age'])
                age_predicts.append(datasets_stats[corpus_name]['train']['majority_class']['age'])

                filenames.append(missing_file)

        res[corpus_name] = (
            {'gen': gen_targets, 'age': age_targets}, 
            {'gen': gen_predicts, 'age': age_predicts}, 
            filenames
        )
    
    return res


def mask_grouping(targets: dict[list], predicts: dict[list], samples_info: dict[tuple], current_phase: str = None, datasets_stats: dict[dict] = None) -> dict[tuple]:
    """Mask grouping
    Args:
        targets (dict[list]): Dict of lists with targets for 'gen' and 'age'
        predicts (dict[list]): Dict of lists with predicts for 'gen' and 'age'
        samples_info (dict[tuple]): List of dicts with sample info
        current_phase (str): Current phase, can be train/dev/test. Defaults to None.
        datasets_stats: Metadata statistics for all corpora. Defaults to None.

    Returns:
        dict[tuple]: For each db tuple: 
                        targets - dict with keys ['gen', 'age', 'mask']
                        predicts - dict with keys ['gen', 'age', 'mask']
                        list(filenames)
    """
    res = {}
    filenames = []
    corpora_names = []
    for s_info in samples_info:
        filenames.extend(s_info['filename'])
        corpora_names.extend(s_info['corpus_name'])
    
    d_gen = {}
    d_gen['predicts'] = predicts['gen']
    d_gen['targets'] = targets['gen']
    d_gen['filenames'] = filenames
    d_gen['corpora_names'] = corpora_names
    df_all_gen = pd.DataFrame(d_gen)

    d_age = {}
    d_age['predicts'] = predicts['age']
    d_age['targets'] = targets['age']
    d_age['filenames'] = filenames
    d_age['corpora_names'] = corpora_names
    df_all_age = pd.DataFrame(d_age)
    
    d_mask = {}
    d_mask['predicts'] = predicts['mask']
    d_mask['targets'] = targets['mask']
    d_mask['filenames'] = filenames
    d_mask['corpora_names'] = corpora_names
    df_all_mask = pd.DataFrame(d_mask)

    ordered_corpora_names = sorted(list(set(corpora_names)))
    for corpus_name in ordered_corpora_names:        
        df_gen = df_all_gen[df_all_gen['corpora_names'] == corpus_name].copy()
        df_gen['predicts'] = df_gen['predicts'].apply(lambda x: np.argmax(x))
        df_gen = df_gen.groupby('filenames', as_index=False).agg(lambda x: pd.Series.mode(x)[0])
        df_gen = df_gen.sort_values('filenames')
        gen_targets = df_gen['targets'].to_list()
        gen_predicts = df_gen['predicts'].to_list()

        df_age = df_all_age[df_all_age['corpora_names'] == corpus_name].copy()
        df_age = df_age.drop('corpora_names', axis=1)
        df_age = df_age.groupby('filenames', as_index=False).mean()
        df_age = df_age.sort_values('filenames')
        
        age_targets = [round(elem, 2) for elem in df_age['targets'].to_list()]
        age_predicts = [round(elem, 2) for elem in df_age['predicts'].to_list()]
        
        df_mask = df_all_mask[df_all_mask['corpora_names'] == corpus_name].copy()
        df_mask['predicts'] = df_mask['predicts'].apply(lambda x: np.argmax(x))
        df_mask = df_mask.groupby('filenames', as_index=False).agg(lambda x: pd.Series.mode(x)[0])
        df_mask = df_mask.sort_values('filenames')
        mask_targets = df_mask['targets'].to_list()
        mask_predicts = df_mask['predicts'].to_list()

        filenames = df_gen['filenames'].to_list()
    
        if datasets_stats:
            current_phase = current_phase.split('_')[0]
            missing_files = set(datasets_stats[corpus_name][current_phase]['fns'].keys()) - set(filenames)
            for missing_file in missing_files:
                gen_targets.append(datasets_stats[corpus_name][current_phase]['fns'][missing_file]['gen'])
                gen_predicts.append(datasets_stats[corpus_name]['train']['majority_class']['gen'])

                age_targets.append(datasets_stats[corpus_name][current_phase]['fns'][missing_file]['age'])
                age_predicts.append(datasets_stats[corpus_name]['train']['majority_class']['age'])
                
                mask_targets.append(datasets_stats[corpus_name][current_phase]['fns'][missing_file]['mask'])
                mask_predicts.append(datasets_stats[corpus_name]['train']['majority_class']['mask'])

                filenames.append(missing_file)

        res[corpus_name] = (
            {'gen': gen_targets, 'age': age_targets, 'mask': mask_targets}, 
            {'gen': gen_predicts, 'age': age_predicts, 'mask': mask_predicts}, 
            filenames
        )
    
    return res

