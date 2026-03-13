import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, List
import pandas as pd
import torch
from sklearn.utils import resample

from preprocess.data_loader import *
from preprocess.preprocess_data import *
from preprocess.rebalance import *
from config.config_data import CONFIG_DATA

__all__ = ["DataManager"]

class DataManager:
    def __init__(self, 
                 input_root: str, 
                 work_dir: str, 
                 config_data: CONFIG_DATA,
                 seed_worker,
                 data_generator: torch.Generator,
                 random_seed: int,
                 tokenizer=None,
                 rebalance=False,
                 train_file: str = "train_ver3.csv",
                 val_file: str = "val_ver3.csv",
                 test_file: str = "test_ver3.csv") -> None:
        
        self.INPUT_ROOT = input_root
        self.WORK_DIR = work_dir
        self.SEED_WORKER = seed_worker
        self.DATA_GENERATOR = data_generator
        self.RANDOM_SEED = random_seed

        self.TARGET_SAMPLES = config_data.TARGET_SAMPLES
        self.MAX_LEN = config_data.MAX_LEN
        self.MAX_SAMPLES_CLASS_0 = config_data.MAX_SAMPLES_CLASS_0
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self._load_raw_csv()
        self._setup_initial_pipeline()
        if rebalance == True:
            self._rebalance()
        self.__create_ml_data()
        self.__create_dataset_multi_task(self.tokenizer)
        self.__create_dataloader_cross_encoder()
        self.__create_dataloader_siamese(self.tokenizer)
        self._all_texts()
        
    def _load_raw_csv(self) -> None:
            self.df_train = pd.read_csv(os.path.join(self.INPUT_ROOT, self.train_file))
            self.df_val = pd.read_csv(os.path.join(self.INPUT_ROOT, self.val_file))
            self.df_test = pd.read_csv(os.path.join(self.INPUT_ROOT, self.test_file))
    def _setup_initial_pipeline(self) -> None:
        all_classes = sorted(list(set(
            self.df_train["Class 1"].tolist() + self.df_train["Class 2"].tolist() +
            self.df_val["Class 1"].tolist() + self.df_val["Class 2"].tolist() +
            self.df_test["Class 1"].tolist() + self.df_test["Class 2"].tolist()
        )))

        self.class_to_token = {c: f"[CLASS_{c}]" for c in all_classes}
        self.class_to_id = {c: i for i, c in enumerate(all_classes)}
        self.NUM_PRODUCT_CLASSES = len(all_classes)
        self._update_data()
    def _all_texts(self) -> None:
        self.all_texts = list(set(
            self.df_train["input_text_1"].tolist() + self.df_train["input_text_2"].tolist() +
            self.df_val["input_text_1"].tolist() + self.df_val["input_text_2"].tolist() +
            self.df_test["input_text_1"].tolist() + self.df_test["input_text_2"].tolist()
        ))
        print(f"Total unique sentences for SimCSE: {len(self.all_texts)}")
    def _update_data(self) -> None:
        self.df_train = preprocess(self.df_train)
        self.df_val = preprocess(self.df_val)
        self.df_test = preprocess(self.df_test)
    
    def _rebalance(self) -> None:
        df_0 = self.df_train[self.df_train['label_score'] == 0]
        df_1 = self.df_train[self.df_train['label_score'] == 1]
        df_2 = self.df_train[self.df_train['label_score'] == 2]
        df_3 = self.df_train[self.df_train['label_score'] == 3]
        df_4 = self.df_train[self.df_train['label_score'] == 4]
        df_0['jaccard'] = df_0.apply(lambda x: calculate_jaccard(x['Term 1'], x['Term 2']), axis=1)
        hard_negatives = df_0[df_0['jaccard'] > 0]
        easy_negatives = df_0[df_0['jaccard'] == 0]
        n_easy_keep = max(0, self.MAX_SAMPLES_CLASS_0 - len(hard_negatives))
        if len(easy_negatives) > n_easy_keep:
            easy_negatives_sampled = resample(easy_negatives, replace=False, n_samples=n_easy_keep, random_state=42)
        else:
            easy_negatives_sampled = easy_negatives

        df_0_final = pd.concat([hard_negatives, easy_negatives_sampled])
        df_0_final = resample(df_0_final, replace=False, n_samples=self.MAX_SAMPLES_CLASS_0, random_state=42)
        
        df_aug_full = augment_cross_pairing(self.df_train)
        df_aug_clean = df_aug_full.drop_duplicates(subset=['input_text_1', 'input_text_2'], keep='first')
        self.df_train = df_aug_clean.copy()
        df_1_final = augment_and_balance(df_1, self.TARGET_SAMPLES)
        df_3_final = augment_and_balance(df_3, self.TARGET_SAMPLES)
        df_2_final = resample(df_2, replace=(len(df_2) < self.TARGET_SAMPLES), n_samples=self.TARGET_SAMPLES, random_state=42)
        df_4_final = resample(df_4, replace=False, n_samples=self.TARGET_SAMPLES, random_state=42)
        self.df_train = pd.concat([df_0_final, df_1_final, df_2_final, df_3_final, df_4_final])
        self.df_train = self.df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    def __create_ml_data(self):
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.vectorizer = create_ml_data(
            self.df_train,
            self.df_val,
            self.df_test
        )

    def __create_patterns(self, tokenizer) -> None:
        self.df_train_aug = create_patterns(self.df_train, tokenizer, self.class_to_token, self.class_to_id)
        self.df_val_aug = create_patterns(self.df_val, tokenizer, self.class_to_token, self.class_to_id)
        self.df_test_aug = create_patterns(self.df_test, tokenizer, self.class_to_token, self.class_to_id)
    def __create_dataset_multi_task(self, tokenizer) -> None:
        self.__create_patterns(tokenizer)
        self.train_ds, self.val_ds, self.test_ds = create_dataset_multi_task(
                self.df_train_aug, self.df_val_aug, self.df_test_aug, self.tokenizer)
    def __create_dataloader_cross_encoder(self) -> None:
        self.train_dataloader, self.evaluator = create_dataloader_cross_encoder(self.df_train, self.df_val)

    def __create_dataloader_siamese(self, tokenizer) -> None:
        self.train_loader, self.val_loader = create_siamese_dataloader(self.df_train, self.df_val, tokenizer)
    def update_siamese_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.__create_dataloader_siamese(self.tokenizer)
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (self.df_train, self.df_val, self.df_test)
    def get_dataloaders(self, model_type=None) -> None:
        if model_type == "siamese":
            return (self.train_loader, self.val_loader)
        elif model_type == "cross_encoder":
            return (self.train_dataloader, self.evaluator)
        else:
            return None
    def get_dataset(self) -> Tuple:
        return (self.train_ds, self.val_ds, self.test_ds)
    def get_ml_data(self) -> Tuple:
        return (self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test)
    def get_all_texts(self) -> List:
        return self.all_texts