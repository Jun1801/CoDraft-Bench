import pandas as pd
def create_dataset(df, tokenizer, class_to_token, class_to_id):
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


