import gc

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, DataCollatorWithPadding

from config import *
from model.models.BiEncoder import get_model_bi_encoder_baseline
from model.models.Xgboost import get_model_xgboost, get_xgboost_sample_weights
from model.models.SVM import get_model_svm
def train_xgboost(X_train, y_train, X_val, y_val, class_weights_array=None):
    model, vectorizer = get_model_xgboost()
    train_args = CONFIG_MODEL.MODEL_CONFIG["xgboost"]["train_args"]
    sample_weights = None
    if class_weights_array is not None:
        sample_weights = get_xgboost_sample_weights(y_train, class_weights_array)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        **train_args 
    )
    
    return model, vectorizer

def train_svm(X_train, y_train):
    model, vectorizer = get_model_svm()
    model.fit(X_train, y_train)    
    return model, vectorizer


def train_bi_encoder_baseline(input_model_path, train_loader, val_loader, class_weights, output_path, device):    
    model = get_model_bi_encoder_baseline(input_model_path, num_classes=CONFIG_MODEL.NUM_CLASSES)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG_MODEL.MODEL_CONFIG['siamese']['lr'])
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    best_f1 = 0.0
    num_epochs = CONFIG_MODEL.MODEL_CONFIG['siamese']['num_epochs_cls']
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in pbar:
            ids1 = batch["ids1"].to(device)
            mask1 = batch["mask1"].to(device)
            ids2 = batch["ids2"].to(device)
            mask2 = batch["mask2"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(ids1, mask1, ids2, mask2)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                ids1 = batch["ids1"].to(device)
                mask1 = batch["mask1"].to(device)
                ids2 = batch["ids2"].to(device)
                mask2 = batch["mask2"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(ids1, mask1, ids2, mask2)
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"Epoch {epoch + 1} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            model.save(output_path)    
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


def train_cross_encoder(model, train_dataloader, evaluator):
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        save_best_model=True,
        optimizer_params={'lr': CONFIG_MODEL.LEARNING_RATE},
        weight_decay=CONFIG_MODEL.WEIGHT_DECAY,
        epochs=CONFIG_MODEL.MODEL_CONFIG['cross_encoder']['epochs'],
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path='./output/nice_model_3prongs_strategy',
        show_progress_bar=True,
        evaluation_steps=500
        )
    return model

def train_multi_task(model, training_args, train_ds, val_ds, tokenizer, compute_metrics):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()
    return trainer
