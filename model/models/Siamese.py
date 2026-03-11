import pandas as pd
import gc
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset

from config import *

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

    model.max_seq_length = CONFIG_DATA.MAX_LEN
    
    clean_sentences = [str(s) for s in sentences if pd.notna(s) and str(s).strip() != ""]
    train_data = [InputExample(texts=[s, s]) for s in clean_sentences]
    
    dataloader = DataLoader(train_data, batch_size=CONFIG_MODEL.BATCH_SIZE, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    print("Starting SimCSE Training...")
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=CONFIG_MODEL.MODEL_CONFIG['siamese']['num_epochs_simcse'],
        warmup_steps=CONFIG_MODEL.MODEL_CONFIG['siamese']['warmup_steps'],
        optimizer_params={'lr': CONFIG_MODEL.LEARNING_RATE},
        show_progress_bar=True,
        output_path=output_path
    )
    print(f"SimCSE Model saved at {output_path}")
    
    del model, train_loss, dataloader
    torch.cuda.empty_cache()
    gc.collect()

# train_simcse(MODEL_NAME, all_texts, PATH_SIMCSE)

class SiameseClassifier(nn.Module):
    def __init__(self, base_model_path, num_classes=5):
        super(SiameseClassifier, self).__init__()
        
        self.encoder = SentenceTransformer(base_model_path, trust_remote_code=True)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension() # 1024
        
        input_dim = self.embedding_dim * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes) 
        )
        
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        out1 = self.encoder({"input_ids": input_ids1, "attention_mask": attention_mask1})
        u = out1["sentence_embedding"]
        
        out2 = self.encoder({"input_ids": input_ids2, "attention_mask": attention_mask2})
        v = out2["sentence_embedding"]
        
        features = torch.cat([u, v, torch.abs(u - v)], dim=1)
        
        logits = self.classifier(features)
        return logits
    
    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, "siamese_state.pth"))
        self.encoder.save(os.path.join(path, "backbone"))
def get_model_siamese():
    pass