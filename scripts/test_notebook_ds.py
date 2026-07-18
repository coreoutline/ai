import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

HF_NAME = "facebook/bart-large-mnli"
tok = AutoTokenizer.from_pretrained(HF_NAME)

data = pd.read_csv(r"C:\Users\tsuma.thomas\Documents\CoreOutline\transformer\data\tool-use-multiturn-expanded.csv")
data['conversation_so_far_str'] = [ " ".join([ j['value'] for j in json.loads(i)]) for i in data['conversation_so_far'] ]
data['tool_descriptions'] = [ [ j['function']['description'] for j in json.loads(i) ] for i in data['tool_list'] ]
data['tool_names'] = [ [ j['function']['name'] for j in json.loads(i) ] for i in data['tool_list'] ]

# Notice that selected_tools is loaded as a string representing a list (or we need json.loads)
print("Type of selected_tools element:", type(data['selected_tools'].iloc[0]))
print("Value:", data['selected_tools'].iloc[0])

class ToolSelectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df
        self.X = df['conversation_so_far_str']
        self.y = df['tool_names']
        self.selected_tools = df['selected_tools']
        self.tool_descriptions = df['tool_descriptions']

    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        # We need to be careful with type of selected_tools
        selected = self.selected_tools.tolist()[idx]
        if isinstance(selected, str):
            selected = json.loads(selected)
        
        _target_classes = list(map(lambda x: 1 if x in selected else 0, self.y.tolist()[idx]))
        _target_enc = [ tok( self.X.tolist()[idx], i, return_tensors="pt", truncation=True, padding="max_length", max_length=512) for i in self.tool_descriptions.tolist()[idx] ]
        return _target_enc, _target_classes

dataset = ToolSelectionDataset(data.head(5), tok)
enc, targets = dataset[0]
print("enc length:", len(enc))
print("first encoding keys:", enc[0].keys())
print("first encoding input_ids shape:", enc[0]['input_ids'].shape)
print("targets:", targets)
