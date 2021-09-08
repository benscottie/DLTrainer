# Libraries
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Dataset Class
class PyTorchDataset(torch.utils.data.Dataset):
  def __init__(self, text_encodings, labels):
    self.text = text_encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {}
    item['text'] = {key: torch.tensor(val[idx]) for key, val in self.text.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item
  
  def __len__(self):
    return len(self.labels)

# Get Dataset
def get_dataset(data_path, model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path) #load tokenizer

    # Load Data
    if data_path:
        df = pd.read_csv(data_path)
        df.dropna(subset=['cleaned_text', 'target'], inplace=True)
    else:
        df = None
    
    text = df['cleaned_text'].values.tolist()
    labels = df['target'].values.tolist()

    text_encodings = tokenizer(text, padding=True, truncation=True) #tokenize text
    
    dataset = PyTorchDataset(text_encodings, labels) #get pytorch dataset
    
    return dataset

# Get DataLoader
def get_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True
    )