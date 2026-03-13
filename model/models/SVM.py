from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

from config import CONFIG_MODEL

def get_model_svm():
    svm_config = CONFIG_MODEL.MODEL_CONFIG["svm"]
    model = SVC(**svm_config["model_args"])
    
    vectorizer = TfidfVectorizer(
        max_features=svm_config["max_features"], 
        lowercase=True
    )
    return model, vectorizer