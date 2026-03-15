import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

import os
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from torch.utils.data import DataLoader

from model.models.BiEncoder import get_model_bi_encoder_baseline, BasicBiEncoderClassifier
from preprocess.data_loader import PairSiameseDataset

from config import *
def get_preds_ml(model, X_test, y_test):
    test_preds = model.predict(X_test)
    return (test_preds, y_test)

def get_preds_multi(trainer, test_ds, df_test):
    test_output = trainer.predict(test_ds)
    predictions = test_output.predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    reshaped_logits = predictions.reshape(-1, 2, 5)
    avg_logits = reshaped_logits.mean(axis=1)

    test_preds = np.argmax(avg_logits, axis=-1)
    test_true = df_test["label_score"].values
    return (test_preds, test_true)
def get_preds_cross_encoder(model, df_test):
    test_inputs = [
        [str(row['input_text_1']), str(row['input_text_2'])]
        for i, row in df_test.iterrows()
    ]
    test_output = model.predict(test_inputs)
    test_preds = np.argmax(test_output, axis=1)
    test_true = df_test["label_score"].values
    return (test_preds, test_true)

def get_preds_siamese(test_df, model_path,model_name, device):
    _, test_preds = _predict_probabilities(model_path,model_name, test_df, device)
    return (test_preds, test_df["label_score"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(labels, tuple):
        labels = labels[0]

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    mae = mean_absolute_error(labels, preds)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "qwk": float(qwk),
        "mae": float(mae)
    }
def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def get_stats(df):
    y_true = df["label"].to_numpy()
    y_pred = df["pred"].to_numpy()

    labels = list(range(5))
    label_names = ['Dissimilar (0)', 'Low (1)', 'Similar (2)', 'High (3)', 'Identical (4)']
    K = len(labels)
    idx = {c: i for i, c in enumerate(labels)}

    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    per_class = []
    for i, c in enumerate(labels):
        support = cm[i, :].sum()
        precision = safe_div(TP[i], TP[i] + FP[i])
        recall = safe_div(TP[i], TP[i] + FN[i])
        f1 = safe_div(2 * precision * recall, precision + recall)
        acc = safe_div(TP[i] + TN[i], TP[i] + TN[i] + FP[i] + FN[i])
        per_class.append({
            "class": c,
            "support": int(support),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    TP_micro = TP.sum()
    FP_micro = FP.sum()
    FN_micro = FN.sum()

    micro_precision = safe_div(TP_micro, TP_micro + FP_micro)
    micro_recall = safe_div(TP_micro, TP_micro + FN_micro)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_precision = np.mean([r["precision"] for r in per_class])
    macro_recall = np.mean([r["recall"] for r in per_class])
    macro_f1 = np.mean([r["f1"] for r in per_class])

    overall_acc = safe_div((y_true == y_pred).sum(), len(y_true))

    mae = np.mean(np.abs(y_true - y_pred))

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    print("Per-class metrics (class | support | accuracy | precision | recall | f1):")
    for r in per_class:
        print(f"{r['class']:>2} | {r['support']:>6} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f}")

    print("\nOverall accuracy:", f"{overall_acc:.4f}")
    print("Micro Precision | Recall | F1:", f"{micro_precision:.4f}", f"{micro_recall:.4f}", f"{micro_f1:.4f}")
    print("Macro Precision | Recall | F1:", f"{macro_precision:.4f}", f"{macro_recall:.4f}", f"{macro_f1:.4f}")

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print("\nConfusion Matrix (Hàng = Thực tế, Cột = Dự đoán):")
    header = "    " + "".join([f" P{c:>3}" for c in labels])
    print(header)
    print("   " + "-" * len(header))
    for i, row in enumerate(cm):
        print(f"T{labels[i]} |" + "".join([f"{val:>4}" for val in row]))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)

    plt.savefig('confusion_matrix.pdf', format='pdf', bbox_inches='tight')

    plt.savefig('confusion_matrix.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()
    return overall_acc, mae, qwk



def save_model(trainer, tokenizer, model_name, save_path):
    print(f"Saving: {save_path} ...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    shutil.make_archive(model_name, 'zip', save_path)
    print("Success!")

def zip_model_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Cannot find: {folder_path}")
        return
    base_name = folder_path.rstrip('/')
    
    print(f"Compressing folder: {folder_path} ...")
    
    try:
        shutil.make_archive(
            base_name=base_name,  
            format='zip',         
            root_dir=folder_path  
        )
        print(f"Sucessfully compressed: {base_name}.zip")
    except Exception as e:
        print(f"Error : {e}")

def _predict_probabilities(model_path,model_name, test_df, device):
    model = get_model_bi_encoder_baseline(
        input_model_path=model_name,
        num_classes=CONFIG_MODEL.NUM_CLASSES
    )
    
    state_dict_path = os.path.join(model_path, "siamese_state.pth")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    tokenizer = model.encoder.tokenizer
    
    test_ds = PairSiameseDataset(test_df, tokenizer, max_len=CONFIG_MODEL.MAX_LEN) 
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=CONFIG_MODEL.MODEL_CONFIG['siamese']['batch_size'], 
        shuffle=False
    )
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting Baseline"):
            ids1 = batch["ids1"].to(device)
            mask1 = batch["mask1"].to(device)
            ids2 = batch["ids2"].to(device)
            mask2 = batch["mask2"].to(device)
            
            logits = model(ids1, mask1, ids2, mask2)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    return np.array(all_probs), np.array(all_preds)