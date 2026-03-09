import torch

class CONFIG_MODEL:
    DEVICE = 'cuda'
    NUM_CLASSES = 5
    NUM_PRODUCT_CLASSES = 45
    MAX_LEN = 256
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MODEL_CONFIG = {
        "cross_encoder": {
            "output_path" : "./output/cross_encoder",
            "evaluation_step" : 500,
            "show_progress_bar" : True,
            "save_best_model" : True,
            "epochs": 5
        },
        "siamese" : {
            "output_path": "./output/siamese"
        },
        "multi_task": {
            "training_args":{
                "output_path": "./output/multi_task",
                "epochs": 10,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 2,
                "per_device_eval_batch_size": 16,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "eval_strategy": "epoch",
                "save_strategy": "epoch",
                "save_total_limit": 1,
                "logging_steps": 10,
                "report_to": "none",
                "load_best_model_at_end": True,
                "metric_for_best_model": "f1_marco", 
                "greater_is_better": True,
                "fp16": torch.cuda.is_available(),
                "seed": 42,
                "data_seed": 42,
                "remove_unused_columns": False
            },
            "loss_args": {
                "alpha": 0.47,
                "aux_weight": 0.26
            }
        }
    }