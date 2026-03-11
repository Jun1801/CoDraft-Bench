import pandas as pd
import gc
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from config import *

class SiameseClassifier(nn.Module):
    def __init__(self, base_model_path, num_classes=5):
        super(SiameseClassifier, self).__init__()
        
        self.encoder = SentenceTransformer(base_model_path, trust_remote_code=True)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension() 
        
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
    
def get_model_siamese(input_model_path, num_classes):
    model = SiameseClassifier(input_model_path, num_classes=num_classes)
    return model
