import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class BasicBiEncoderClassifier(nn.Module):
    def __init__(self, base_model_path, num_classes=5):
        super(BasicBiEncoderClassifier, self).__init__()
        
        self.encoder = SentenceTransformer(base_model_path, trust_remote_code=True)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension() 
        input_dim = self.embedding_dim * 3
        hidden_dim = self.embedding_dim 
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
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
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "siamese_state.pth"))
        self.encoder.save(os.path.join(path, "backbone"))

def get_model_bi_encoder_baseline(input_model_path, num_classes):
    model = BasicBiEncoderClassifier(input_model_path, num_classes=num_classes)
    return model