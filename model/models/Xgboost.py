import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from config import CONFIG_MODEL

def get_model_xgboost():
    xgb_config = CONFIG_MODEL.MODEL_CONFIG["xgboost"]
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=CONFIG_MODEL.NUM_CLASSES,
        device=CONFIG_MODEL.DEVICE,
        tree_method='hist',
        **xgb_config["model_args"]
    )
    vectorizer = TfidfVectorizer(
        max_features=xgb_config["max_features"], 
        lowercase=True
    )
    return model, vectorizer

def get_xgboost_sample_weights(y_train, class_weights_array):
    return np.array([class_weights_array[int(label)] for label in y_train])

