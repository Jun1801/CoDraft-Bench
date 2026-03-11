import gc
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import Trainer, DataCollatorWithPadding

from config import *
from model.models.Siamese import SiameseClassifier
from preprocess.data_loader import PairSiameseDataset
def train_simcse(model_name, sentences, output_path, device):
    print("STEP 1: Training SimCSE")
    
    model = SentenceTransformer(
        model_name, 
        trust_remote_code=True,
        device=device,
        config_kwargs={
            "use_memory_efficient_attention": False,
            "unpad_inputs": False
        }
    )

    model.max_seq_length = 256
    
    clean_sentences = [str(s) for s in sentences if pd.notna(s) and str(s).strip() != ""]
    train_data = [InputExample(texts=[s, s]) for s in clean_sentences]
    
    dataloader = DataLoader(train_data, batch_size=CONFIG_MODEL.BATCH_SIZE, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    print("Starting SimCSE Training...")
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=CONFIG_MODEL.MODEL_CONFIG['siamese']['num_epochs_simcse'],
        warmup_steps=CONFIG_MODEL.MODEL_CONFIG['siamese']['warmup_steps'],
        optimizer_params={'lr': CONFIG_MODEL.MODEL_CONFIG['siamese']['lr']},
        show_progress_bar=True,
        output_path=output_path
    )
    print(f"SimCSE Model saved at {output_path}")
    
    del model, train_loss, dataloader
    torch.cuda.empty_cache()
    gc.collect()

def train_siamese(input_model_path, train_loader, val_loader, class_weights, output_path, device):
    print("STEP 2: Training Similarity Classification")
    
    torch.cuda.empty_cache()
    gc.collect()  
    
    model = SiameseClassifier(input_model_path, num_classes=5)
    
    try:
        model.encoder[0].auto_model.gradient_checkpointing_enable()
        print("Gradient Checkpointing ENABLED")
    except Exception as e:
        print(f"Cannot turn on Gradient Checkpointing: {e}")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG_MODEL.MODEL_CONFIG['siamese']['lr'])
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    scaler = GradScaler("cuda")
    best_f1 = 0.0
    
    for epoch in range():
        model.train()
        total_loss = 0
        optimizer.zero_grad() 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG_MODEL.MODEL_CONFIG['siamese']['num_epochs_cls']}")
        
        for i, batch in enumerate(pbar):
            ids1 = batch["ids1"].to(device)
            mask1 = batch["mask1"].to(device)
            ids2 = batch["ids2"].to(device)
            mask2 = batch["mask2"].to(device)
            labels = batch["label"].to(device)
            
            with autocast("cuda"): 
                logits = model(ids1, mask1, ids2, mask2)
                loss = criterion(logits, labels)
                
                loss = loss / CONFIG_MODEL.MODEL_CONFIG['siamese']['accumulation_step']
            
            scaler.scale(loss).backward()
            
            if (i + 1) % CONFIG_MODEL.MODEL_CONFIG['siamese']['accumulation_step'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            current_loss = loss.item() * CONFIG_MODEL.MODEL_CONFIG['siamese']['accumulation_step']
            total_loss += current_loss
            pbar.set_postfix({"loss": current_loss})
            
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
                
                with autocast("cuda"):
                    logits = model(ids1, mask1, ids2, mask2)
                
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"Epoch {epoch + 1} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            print(f"Saving Best Model to {output_path}...")
            os.makedirs(output_path, exist_ok=True)
            model.save(output_path)
            
            torch.cuda.empty_cache()
            
    print("Training Complete")
    del model, optimizer, train_loader, val_loader, scaler
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
