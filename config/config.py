import tiktoken
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

tokenizer = tiktoken.get_encoding("gpt2")
with open("../../data/verdict.txt", "r", encoding="utf-8") as f:
    txt = f.read()
    txt = txt.replace("<|endoftext|>", " <|endoftext|> ")
    token_ids = tokenizer.encode(txt)
print(len(tokenizer._mergeable_ranks))
CORE_TRANSFORMER_CONFIG = {
    "vocab_size": len(tokenizer._mergeable_ranks), # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads":12, # Number of attention heads
    "n_layers": tune.choice([12, 24, 36, 48]), # Number of layers
    "drop_rate": 0.1,#tune.choice([0.1,0.5]), # Dropout rate
    "qkv_bias": False, # Query-Key-Value bias
    "lr":1e-4 # tune.loguniform(1e-4, 1e-1) # Learning rate
}