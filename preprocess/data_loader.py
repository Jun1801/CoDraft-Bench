import pandas as pd
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from datasets import Dataset

def create_patterns(df, tokenizer, class_to_token, class_to_id):
    new_rows = []
    mask_token = tokenizer.mask_token

    for _, row in df.iterrows():
        c1, t1 = row["Class 1"], row["input_text_1"]
        c2, t2 = row["Class 2"], row["input_text_2"]
        label = row["label_score"]

        text1_a = f"{mask_token} {t1}"
        text2_a = f"{class_to_token.get(c2, '')} {t2}"
        label_aux_a = class_to_id.get(c1, 0)

        new_rows.append({
            "text1": text1_a, "text2": text2_a,
            "labels": label, "aux_labels": label_aux_a
        })

        text1_b = f"{class_to_token.get(c1, '')} {t1}"
        text2_b = f"{mask_token} {t2}"
        label_aux_b = class_to_id.get(c2, 0)

        new_rows.append({
            "text1": text1_b, "text2": text2_b,
            "labels": label, "aux_labels": label_aux_b
        })

    return pd.DataFrame(new_rows)

def preprocess_dataset(examples, tokenizer):
    tokenized = tokenizer(
        examples["text1"],
        examples["text2"],
        truncation=True,
        max_length=192,
        padding=False
    )
    tokenized["labels"] = examples["labels"]
    tokenized["aux_labels"] = examples["aux_labels"]
    return tokenized

def create_dataloader_cross_encoder(df_train, df_val):
    train_samples = []
    for i, row in df_train.iterrows():
    
        train_samples.append(InputExample(
            texts=[str(row['input_text_1']), str(row['input_text_2'])],
            label= int(row['label_score'])
        ))
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
    val_samples = []
    for i, row in df_val.iterrows():
        val_samples.append(InputExample(
            texts=[str(row['input_text_1']), str(row['input_text_2'])],
            label= int(row['label_score'])
        ))

    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(
        val_samples,
        name='Ordinal_Check'
    )
    return train_dataloader, evaluator

def create_dataset_multi_task(df_train_aug, df_val_aug, df_test_aug):
    cols_to_remove = df_train_aug.columns.tolist()

    train_ds = Dataset.from_pandas(df_train_aug).map(
        preprocess_dataset,
        batched=True,
        remove_columns=cols_to_remove
    )

    val_ds = Dataset.from_pandas(df_val_aug).map(
        preprocess_dataset,
        batched=True,
        remove_columns=df_val_aug.columns.tolist()
    )

    test_ds = Dataset.from_pandas(df_test_aug).map(
        preprocess_dataset,
        batched=True,
        remove_columns=df_test_aug.columns.tolist()
    )
    cols = ["input_ids", "attention_mask", "labels", "aux_labels"]
    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format(type="torch", columns=cols)
    return train_ds, val_ds, test_ds