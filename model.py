# Libraries
from transformers import AutoConfig, AutoModelForSequenceClassification

# Load Model
def get_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    return AutoModelForSequenceClassification.from_config(config)