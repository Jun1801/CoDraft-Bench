import pandas as pd
import numpy as np
from .preprocess_data import create_structured_text_enhanced
def calculate_jaccard(text1, text2):
    set1 = set(str(text1).lower().split())
    set2 = set(str(text2).lower().split())
    if len(set1) == 0 or len(set2) == 0: return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union


def augment_cross_pairing(df_input, target_labels=[3, 4]):
    df = df_input.copy()
    def make_key_series(c, t): return c.astype(str) + " | " + t.astype(str)
    
    df['key_1'] = make_key_series(df['Class 1'], df['Term 1'])
    df['key_2'] = make_key_series(df['Class 2'], df['Term 2'])
    
    existing_hashes = set(np.where(
        df['key_1'] < df['key_2'], 
        df['key_1'] + "___" + df['key_2'], 
        df['key_2'] + "___" + df['key_1']
    ))
    
    high_quality_df = df[df['label_score'].isin(target_labels)].copy()

    cols_to_keep = ['label_score', 'Class 2', 'Term 2', 'Nature 2', 'Purpose 2']
    rename_fwd = {'key_1': 'parent', 'key_2': 'child', 'label_score': 'score', 'Class 2': 'class_child', 'Term 2': 'term_child', 'Nature 2': 'nature_child', 'Purpose 2': 'purpose_child'}
    
    forward = high_quality_df[['key_1', 'key_2'] + cols_to_keep].rename(columns=rename_fwd)
    
    cols_to_keep_bwd = ['label_score', 'Class 1', 'Term 1', 'Nature 1', 'Purpose 1']
    rename_bwd = {'key_2': 'parent', 'key_1': 'child', 'label_score': 'score', 'Class 1': 'class_child', 'Term 1': 'term_child', 'Nature 1': 'nature_child', 'Purpose 1': 'purpose_child'}
    
    backward = high_quality_df[['key_2', 'key_1'] + cols_to_keep_bwd].rename(columns=rename_bwd)
    
    connections = pd.concat([forward, backward]).drop_duplicates(subset=['parent', 'child'])
    merged = pd.merge(connections, connections, on='parent', suffixes=('_1', '_2'))
    
    new_pairs = merged[merged['child_1'] < merged['child_2']].copy()
    if len(new_pairs) == 0: return df_input
        
    new_pairs['label_score'] = new_pairs[['score_1', 'score_2']].min(axis=1)

    df_aug = pd.DataFrame({
        "label_score": new_pairs["label_score"],
        "Class 1": new_pairs["class_child_1"], "Term 1": new_pairs["term_child_1"],
        "Nature 1": new_pairs["nature_child_1"], "Purpose 1": new_pairs["purpose_child_1"],
        
        "Class 2": new_pairs['class_child_2'], "Term 2": new_pairs['term_child_2'],
        "Nature 2": new_pairs["nature_child_2"], "Purpose 2": new_pairs["purpose_child_2"],
    })

    df_aug['key_1'] = make_key_series(df_aug['Class 1'], df_aug['Term 1'])
    df_aug['key_2'] = make_key_series(df_aug['Class 2'], df_aug['Term 2'])
    
    df_aug['pair_hash'] = np.where(
        df_aug['key_1'] < df_aug['key_2'], 
        df_aug['key_1'] + "___" + df_aug['key_2'], 
        df_aug['key_2'] + "___" + df_aug['key_1']
    )
    
    df_aug = df_aug[~df_aug['pair_hash'].isin(existing_hashes)].copy()
    
    if len(df_aug) > 0:
        df_aug['input_text_1'] = df_aug.apply(
            lambda x: create_structured_text_enhanced(x['Term 1'], x['Nature 1'], x['Purpose 1'], x['Class 1']), axis=1
        )
        df_aug['input_text_2'] = df_aug.apply(
            lambda x: create_structured_text_enhanced(x['Term 2'], x['Nature 2'], x['Purpose 2'], x['Class 2']), axis=1
        )

    print(f"Number of new cross_pairing: {len(df_aug)}")
    
    cols_to_drop = ['key_1', 'key_2', 'pair_hash']
    return pd.concat([df_input, df_aug.drop(columns=cols_to_drop, errors='ignore')], ignore_index=True)

def augment_and_balance(df_sub, target_count):
    df_flipped = df_sub.copy()
    df_flipped['input_text_1'] = df_sub['input_text_2']
    df_flipped['input_text_2'] = df_sub['input_text_1']
    df_flipped['Class 1'] = df_sub['Class 2']
    df_flipped['Class 2'] = df_sub['Class 1']
    df_flipped['Term 1'] = df_sub['Term 2']
    df_flipped['Term 2'] = df_sub['Term 1']
    
    df_flipped['Nature 1'] = df_sub['Nature 2']
    df_flipped['Nature 2'] = df_sub['Nature 1']
    
    df_flipped['Purpose 1'] = df_sub['Purpose 2']
    df_flipped['Purpose 2'] = df_sub['Purpose 1']

    df_aug = pd.concat([df_sub, df_flipped]).drop_duplicates(subset=['input_text_1', 'input_text_2'])

    if len(df_aug) < target_count:
        return resample(df_aug, replace=True, n_samples=target_count, random_state=42)
    elif len(df_aug) > target_count:
        return resample(df_aug, replace=False, n_samples=target_count, random_state=42)
    return df_aug