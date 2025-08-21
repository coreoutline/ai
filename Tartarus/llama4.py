import math
import torch
import torch.nn as nn

import torch.nn.functional as F
# import wandb

from torch.utils.data import Dataset, DataLoader
import tiktoken
from torch.utils.data import random_split

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt


class CoreDataset(Dataset):
    def __init__(self, txt, tokenizer, max_len, step):
        device = torch.device("cuda")

        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_len, step):
            input_chunk = token_ids[i:i+max_len]
            target_chunk = token_ids[i+1: i+max_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        self.input_ids = torch.stack(self.input_ids)
        self.target_ids = torch.stack(self.target_ids)
        self.input_ids.to(device)
        self.target_ids.to(device)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return { "input_ids" : self.input_ids[index], "labels": self.input_ids[index], "attention_mask": torch.ones_like(self.input_ids[index]) }
    

def create_dataloader_v1(txt, batch_size=4, max_len=256,step=128, shuffle=True, drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = CoreDataset(txt, tokenizer, max_len, step)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader, tokenizer


# with open("/Users/apple/Documents/CoreOutline/transformer/data/content.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
#     raw_text = raw_text.replace("<|endoftext|>", " <|endoftext|> ")
#     dataloader, tokenizer = create_dataloader_v1( raw_text, batch_size=1, max_len=100, step=1, shuffle=False)
#     total_size = len(dataloader.dataset)
#     train_size = int(0.8 * total_size)
#     val_size = int(0.1 * total_size)
#     test_size = total_size - train_size - val_size

#     train_dataset, val_dataset, test_dataset = random_split(dataloader.dataset, [train_size, val_size, test_size])

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


from torch.utils.data import Dataset, DataLoader
import torch
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Train a BPE tokenizer
def train_bpe_tokenizer(data_files, vocab_size=32000, model_prefix="llama_tokenizer"):
    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Pre-tokenize using whitespace
    tokenizer.pre_tokenizer = Whitespace()
    
    # Initialize trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=2
    )
    
    # Prepare iterator over files
    def batch_iterator():
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()
    
    # Train the tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Save the tokenizer
    tokenizer.save(f"{model_prefix}.json")
    
    return tokenizer

# Create a Dataset class
class TextDataset(Dataset):
    def __init__(self, data_files, tokenizer, max_length=1024, stride=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []
        
        # Load and tokenize data
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Process large text in chunks to avoid memory issues
            chunk_size = 100000  # Process 100K characters at a time
            all_tokens = []
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                # Encode using BPE tokenizer
                encoded = self.tokenizer.encode(chunk)
                all_tokens.extend(encoded.ids)
            
            # Create examples with stride
            for i in range(0, len(all_tokens) - max_length + 1, stride):
                self.examples.append(all_tokens[i:i + max_length])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Prepare input_ids and labels
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # Create attention mask (all 1s for full sequences)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
# Define your data files
data_files = ["C://Users/tsuma.thomas/Documents/CoreOutline/transformer/data/content.txt"]

# Train BPE tokenizer
tokenizer = train_bpe_tokenizer(data_files, vocab_size=32000)

# Create dataset
dataset = TextDataset(data_files, tokenizer, max_length=1024, stride=512)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Check the first batch
batch = next(iter(dataloader))
print(f"Input shape: {batch['input_ids'].shape}")
print(f"Label shape: {batch['labels'].shape}")

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True).clamp(min=self.eps))
        x_norm = x / rms * self.weight
        return x_norm

def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    """Apply rotary embeddings to input tensors."""
    # Reshape last dimension: [d] -> [d/2, 2]
    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)
    
    # Convert to complex numbers
    x_complex = torch.view_as_complex(x_reshape)
    
    # Reshape freqs_cis for proper broadcasting
    # If x is [batch, seq, heads, dim], freqs_cis should be [1, seq, 1, dim/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation through complex multiplication
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real and reshape
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)  
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis[:seq_len])
        k = apply_rotary_emb(k, freqs_cis[:seq_len])
        
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(2, 3)) / scale  # FIX: use division, not exponentiation

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, -1)

        out = self.o_proj(out)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256, dropout=0.0):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of -1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        x = swish * self.w3(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, ffn_dim_multiplier=4, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attention = SelfAttention(dim, num_heads, head_dim, dropout)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, int(dim * ffn_dim_multiplier), dropout=dropout)

    def forward(self, x, freqs_cis, mask=None):
        h = x + self.attention(self.attn_norm(x), freqs_cis, mask)

        out = h + self.ffn(self.ffn_norm(h))
        return out

class Erebus(nn.Module):
    def __init__(self, vocab_size, dim=4096,num_layers=32, num_heads=32, max_seq_len=2048, ffn_dim_multiplier=4,dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        head_dim = dim // num_heads

        self.token_embedding = nn.Embedding(vocab_size, dim)

        self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2)

        self.layers = nn.ModuleList([
            TransformerBlock( dim, num_heads=num_heads, head_dim=head_dim, ffn_dim_multiplier=ffn_dim_multiplier,dropout=dropout)
            for _ in  range (num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

        # FIX: correct weight tying
        self.output.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        causal_mask = torch.triu(
            torch.full((seq_len ,seq_len), float('-inf'), device=input_ids.device),
            diagonal=1
        )

        if attention_mask is not None:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            causal_mask = causal_mask + (1.0 - attention_mask) * torch.finfo(torch.float32).min
        else:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        print("causal_mask min/max:", causal_mask.min(), causal_mask.max())
        x = self.token_embedding(input_ids)

        freqs_cis = self.freqs_cis.to(x.device)

        for layer in self.layers:
            x = layer(x, freqs_cis, causal_mask)

        x = self.norm(x)
        print("Shape before output layer:", x.shape)  # Add this line
        logits = self.output(x)

        return logits
    
def train_erebus_model(model, train_data_loader, val_data_loader, optimizer, scheduler, num_epochs, device):
    model.to(device)
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_data_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model.vocab_size),
                shift_labels.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_data_loader)
        train_loss_history.append(avg_train_loss)
        model.eval()
        val_loss = 0.0
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        with torch.no_grad():
            for batch in val_data_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                logits = model(input_ids, attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, model.vocab_size),
                    shift_labels.view(-1)
                )
                val_loss += loss.item()
        avg_train_loss = train_loss / len(train_data_loader)
        avg_val_loss = val_loss / len(val_data_loader)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        torch.save(model.state_dict(), f"erebus_epoch_{epoch+1}.pt")
        # wandb.save(f"erebus_epoch_{epoch+1}.pt")
    # wandb.finish()
import gc


def train_with_mixed_precision(model, train_data_loader, val_data_loader, num_epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loader) * num_epochs)
    scaler = GradScaler()
    model.to(device)
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        for batch_idx,batch in enumerate(train_data_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Clear gradients
            optimizer.zero_grad()
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with autocast():
                logits = model(input_ids, attention_mask)
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
                loss = F.cross_entropy(shift_logits.view(-1, model.vocab_size), shift_labels.view(-1))
                print("train loss: ",loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            torch.save(model.state_dict(), f"erebus_chechpoint_mixed_precision.pt")
        avg_train_loss = train_loss / len(train_data_loader)
        train_loss_history.append(avg_train_loss)
        with torch.no_grad():
            for batch in val_data_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                logits = model(input_ids, attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, model.vocab_size),
                    shift_labels.view(-1)
                )
                val_loss += loss.item()
                # Log validation loss to wandb after each batch
                # wandb.log({"validation_loss": loss.item(), "epoch": epoch + 1})
        avg_train_loss = train_loss / len(train_data_loader)
        avg_val_loss = val_loss / len(val_data_loader)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # Plot and save the loss curves
        plt.clf()
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_loss_history, label='Train Loss')
        plt.plot(range(1, num_epochs+1), val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('erebus_loss_plot_epoch.png')
        plt.close()

def create_erebus_tiny(vocab_size, max_seq_len=512):
    model = Erebus(
        vocab_size=vocab_size,
        dim=384,
        num_layers=6,
        num_heads=6,
        max_seq_len=max_seq_len,
        ffn_dim_multiplier=2.667,
        dropout=0.1
    )
    model.load_state_dict(torch.load("./erebus_chechpoint_mixed_precision.pt"))
    return model

def create_erebus_small(vocab_size, max_seq_len=1024):
    model = Erebus(
        vocab_size=vocab_size,
        dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=max_seq_len,
        ffn_dim_multiplier=2.667,
        dropout=0.1
    )
    return model

device = torch.device("cuda")

model = create_erebus_tiny(vocab_size=526000)

num_epochs = 3

train_with_mixed_precision(model=model, train_data_loader=dataloader, val_data_loader=dataloader, num_epochs=2, device=device)

# optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)

# train_erebus_model(model=model, train_data_loader=train_loader, val_data_loader=val_loader, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs, device=device)


def predict_with_top_k(input_sentence, model, top_k=50, penalty=1.0):
    """ Predict the next part of a sequence using top-k sampling with repetition penalty.
    Args:
        input_sentence (str): The input text to continue
        model: The language model to use for prediction
        top_k (int): Number of highest probability tokens to consider for sampling
        penalty (float): Penalty factor to apply to repeated tokens (1.0 means no penalty)
    Returns:
        str: The generated continuation of the input text
    """
    # Get the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Tokenize the input sentence
    input_tokens = tokenizer.encode(input_sentence)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Track generated tokens for repetition penalty
    generated_tokens = []
    generated_text = input_sentence
    
    # Generate up to 100 new tokens (can be adjusted as needed)
    max_new_tokens = 100
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = input_tensor[:, -512:]
            # Get model predictions
            with torch.no_grad():
                logits = model(idx_cond)
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty if penalty != 1.0 and len(generated_tokens) > 0
            if penalty != 1.0 and len(generated_tokens) > 0:
                for token in set(generated_tokens):
                    next_token_logits[0, token] /= penalty
            
            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            
            # Sample from the probability distribution
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]
            
            # Add the token to the list of generated tokens
            generated_tokens.append(next_token.item())
            
            # Prepare input for next iteration (append the new token)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
            # Decode the token and add it to the generated text
            next_token_text = tokenizer.decode([next_token.item()])
            generated_text += next_token_text
            
            # Check for end of text token or newline
            # if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
            #     break
                
    return generated_text


# print(predict_with_top_k("Churn rate", model, top_k=50, penalty=1.0))