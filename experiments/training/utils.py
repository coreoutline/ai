import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

def generate(model, idx, max_new_tokens, context_size, temperature=0.3, top_k=2, eos_id=None, repetition_penalty=1.2):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)['logits']
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


def format_data(entry):
    """Format data for training."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['prompts'].strip().replace('<prompt>', '').replace('</prompt>', '')}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input'].strip()}" if "input" in entry and entry['input'] else ""
    )
    desired_response = f"\n\n### Response: \n{entry['answers'].strip().replace('<ans>', '').replace('</ans>', '')}"
    return instruction_text + input_text + desired_response

def format_input(entry):
    """Format input for inference."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['prompts'].strip().replace('<prompt>', '').replace('</prompt>', '')}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input'].strip()}" if "input" in entry and entry['input'] else ""
    )
    return instruction_text + input_text

# Alias for backwards compatibility
formatData = format_data

class InstructionDatasetDDP(Dataset):
    """Dataset for instruction tuning using DDP optimized format."""
    def __init__(self, data, tokenizer, max_seq_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        entry = self.data.iloc[index]
        full_text = format_data(entry)
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0).clone()
        }

    def __len__(self):
        return len(self.data)

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed training")
        return None, None, None, None

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank, dist.get_world_size()

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
