import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from evaluate import evaluate_model, generate_and_print_sample, calc_loss_batch
from model import CoreModel
from preprocess_transformer_data import create_dataloader_v1
import time

from psutil import virtual_memory
import multiprocessing

import time
start_time = time.time()
torch.manual_seed(123)
# from config import CORE_TRANSFORMER_CONFIG
from torch.utils.data import random_split
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from huggingface_hub import notebook_login, login


cpu_count = multiprocessing.cpu_count()

mem = virtual_memory()
mem = round(mem.total / 1000000000, 1)


df = pd.read_parquet("hf://datasets/core-outline/nyx-finance-instruct/data/train-00000-of-00001.parquet")
df = df.iloc[0:50]

def formatData(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n ### Instruction:\n{entry['prompts'].strip().replace('<prompt>','').replace('</prompt>','')}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input'].strip()}" if "input" in entry else ""
    )

    desired_response = f"\n\n### Response: \n{entry['answers'].strip().replace('<ans>','').replace('</ans>','')}"

    return instruction_text + input_text + desired_response

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n ### Instruction:\n{entry['prompts'].strip().replace('<prompt>','').replace('</prompt>','')}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input'].strip()}" if "input" in entry else ""
    )


    return instruction_text + input_text + "\n\n### Response: \n"
    
df['llm_input'] = df.apply(formatData, axis=1)

train_portion = int(len(df) * 0.85 )
test_portion = int(len(df) * 0.1 )
val_portion = 0.05

train_data = df.iloc[:train_portion]
test_data = df.iloc[train_portion: train_portion+test_portion]
val_data = df.iloc[train_portion+test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

class InstructionDataset(Dataset):
    def __init__(self,data, tokenizer):
        self.data = data,
        self.encoded_texts = []
        for index,entry in data.iterrows():
            full_text = formatData(entry)
            self.encoded_texts.append(tokenizer.encode(full_text))
    def __getitem__(self,index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

def custom_collate_function( batch, pad_token_id = 50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = ( new_item + [pad_token_id] * (batch_max_length - len(new_item) ))
        
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
    
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
            
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from functools import partial
customized_collate_fn = partial(
    custom_collate_function,
    device=device,
    allowed_max_length=1024
)

num_workers = 4
batch_size = 8

torch.manual_seed(123)


train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn = customized_collate_fn,
    shuffle=True,
    drop_last= True,
    num_workers = num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn = customized_collate_fn,
    shuffle=True,
    drop_last= True,
    num_workers = num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn = customized_collate_fn,
    shuffle=True,
    drop_last= True,
    num_workers = num_workers
)

CONFIG={
    "vocab_size": 50256, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads":12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.0, # Dropout rate
    "qkv_bias": False, # Query-Key-Value bias
    "lr": 6e-5 # Learning rate
}

model =  CoreModel(CONFIG)
model_state_dict = torch.load(f"./models/nyx.pth")
model.load_state_dict(model_state_dict)

model.eval()
torch.manual_seed(123)
# input_text = format_input(val_data.iloc[1])
# print(input_text)


def generate(model, idx, max_new_tokens, context_size, temperature=0.3, top_k=2, eos_id=None, repetition_penalty=1.2):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(idx_cond.shape[1]):
                previous_token = idx_cond[0, i]
                logits[0, previous_token] = logits[0, previous_token] / repetition_penalty

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def load_teacher(model_name="google/gemma-2b-it"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def train(config, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    teacher_model, teacher_tokenizer = load_teacher()
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    global_step = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            queries = batch["llm_input"].to(device)
            print(queries)
            batch_size = len(queries)
            batch_loss = 0.0

            for i, query in enumerate(queries):
                token_ids = generate(
                    model=model,
                    idx=text_to_token_ids(query, tokenizer),
                    max_new_tokens=50,
                    context_size=CONFIG["context_length"],
                    eos_id=50256,
                )

                generated_text = token_ids_to_text(token_ids, tokenizer)
                print(generated_text)

            break
        break

train(CONFIG, epochs=1)