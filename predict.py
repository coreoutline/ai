import torch
import tiktoken
from model import CoreModel
import time
import ollama

client = ollama.Client()




def format_input(entry):
    print(entry)
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n ### Instruction:\n{entry['prompt'].strip().replace('<prompt>','').replace('</prompt>','')}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input'].strip()}" if "input" in entry else ""
    )


    return instruction_text + input_text 


    
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


num_workers = 4
batch_size = 8

torch.manual_seed(123)



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
    print(tokenizer.decode(flat.tolist()))
    response = client.generate(model="llama3.2", prompt=f"""
    The following is the output of a trained small language model. I want you to edit it cleaning up any issues like unnecessary text, repetitions , or illogical text. Also correct the grammar. While doing the cleanup, take note of the instruction to decide how to clean the response. Your answer should be in this format: `Below is an instruction that describes a task. Write a response that appropriately completes the request. ###Instruction <instruction as it was originally with no edit> ###Response: <cleaned_response>`. Do not include any other text. Here is the output to be cleaned: {tokenizer.decode(flat.tolist())}
                               """).response
    return response


def predict(text):
    input_text = format_input({"prompt":text})
    print(input_text)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=200,
        context_size=CONFIG["context_length"],
        eos_id=50256,
    )
    print("Tokens: ",token_ids)

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print("Generated answer: ",generated_text)
    return generated_text


predict("What metrics are available on Core&Outline?")