import numpy as np
import torch.nn as nn
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

def get_model_cross_encoder(model_name, num_classes, max_len, weights_tensor):
    model = CrossEncoder(
        model_name,
        num_labels=num_classes,
        max_length=max_len,
        automodel_args={"ignore_mismatched_sizes": True}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.model.resize_token_embeddings(len(tokenizer))
    model.loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
    return model, tokenizer