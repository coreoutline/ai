from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import torch
from transformers import AutoTokenizer
from src.models.model_2 import core_model
import time
import json
import asyncio
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
load_dotenv()
torch.set_num_threads(torch.get_num_interop_threads()) 

app = FastAPI(title="Transformer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Content-Length", "X-Content-Length", "Content-Range"],
)

@app.options("/predict/stream")
async def options_predict_stream():
    """Handle preflight requests"""
    return {"message": "OK"}

# Initialize tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")

# core_model.load_state_dict(torch.load(f"./models/nyx_2_reasoning.pth"))
# core_model.load_state_dict(torch.load(f"../transformers/nyx_2_reasoning.pth"))
core_model.load_state_dict(torch.load(f"../../models/nyx_2.pth"))
core_model.eval()
core_model.to(device)

def format_input(entry):
    instruction_text = (
        f"You are an expert financial analyst / accountant / data analyst / business intelligence analyst "
        f"You are required to reason through and answer customer questions providing them with answers to their questions. Here is the instruction"
        f"\n\n ### Instruction:\n{entry['prompt'].strip().replace('<prompt>','').replace('</prompt>','')}"
    )
    input_text = (
        f"Here is the context through which you will base your answer on"
        f"\n\n### Input: \n{entry['context'].strip()}" if "context" in entry and entry['context'] else ""
    )
    reasoning_step = (
        "Based on the instruction and the context, here are the reasoning steps that should be taken: \n"
        f"\n\n### Response:\n"
    )
    return instruction_text + input_text + reasoning_step

# def generate_stream(model, idx, max_new_tokens, context_size, temperature=0.1, top_k=3, top_p=0.9, eos_id=None, repetition_penalty=1.2):
#     for _ in range(max_new_tokens):
#         idx_cond = idx[:, -context_size:]
#         with torch.no_grad():
#             logits = model(idx_cond)['logits']
#         logits = logits[:, -1, :]

#         # Apply repetition penalty - optimized to only penalize tokens in the recent context
#         if repetition_penalty != 1.0 and repetition_penalty > 0:
#             # Limit the context for repetition penalty to prevent memory issues
#             # Only consider the last 50 tokens to avoid CUDA memory issues
#             penalty_context = min(50, idx_cond.shape[1])
#             for i in range(max(0, idx_cond.shape[1] - penalty_context), idx_cond.shape[1]):
#                 previous_token = idx_cond[0, i].item()  # Convert to scalar to avoid indexing issues
#                 if previous_token < logits.shape[-1]:  # Ensure token is within vocabulary range
#                     logits[0, previous_token] = logits[0, previous_token] / repetition_penalty

#         # Apply top-p (nucleus) sampling along with top-k for better quality
#         if top_p is not None:
#             # Sort logits and compute cumulative probabilities
#             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#             cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
#             # Remove tokens with cumulative probability above the threshold
#             sorted_indices_to_remove = cumulative_probs > top_p
#             # Shift the indices to the right to keep the first token that exceeds the threshold
#             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#             sorted_indices_to_remove[..., 0] = 0
            
#             # Fill the logits tensor with -infinity for tokens to be removed
#             indices_to_remove = sorted_indices[sorted_indices_to_remove]
#             logits[0, indices_to_remove] = float('-inf')
        
#         if top_k is not None:
#             top_logits, _ = torch.topk(logits, top_k)
#             min_val = top_logits[:, -1]
#             logits = torch.where(
#                 logits < min_val,
#                 torch.tensor(float('-inf')).to(logits.device),
#                 logits
#             )
            
#         # Lower temperature for more focused, coherent responses
#         if temperature > 0.0:
#             logits = logits / temperature
#             probs = torch.softmax(logits, dim=-1)
#             idx_next = torch.multinomial(probs, num_samples=1)
#         else:
#             idx_next = torch.argmax(logits, dim=-1, keepdim=True)

#         if eos_id is not None and idx_next == eos_id:
#             break

#         idx = torch.cat((idx, idx_next), dim=1)
#         token = tokenizer.decode(idx_next[0])
#         yield token


def generate_stream(model, idx, max_new_tokens, context_size, temperature=0.3, top_k=2, eos_id=None, repetition_penalty=1.2):
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
        token = tokenizer.decode(idx_next[0])
        yield token
        # idx = torch.cat((idx, idx_next), dim=1)
    # return idx

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Server is running"}

@app.post("/predict")
async def predict(request: Request):
    """Non-streaming prediction endpoint"""
    data = await request.json()
    text = data.get("prompt", "")
    
    input_text = format_input({"prompt": text})
    inputs = tokenizer(input_text, return_tensors="pt")
    
    token_ids = generate_stream(
        model=core_model,
        idx=inputs['input_ids'].to(device),
        max_new_tokens=800,
        context_size=5012,
        eos_id=50256,
    )
    
    # Collect all tokens into a single response
    response = "".join(token_ids)
    return {"response": response}

# text = format_input({"prompt": "What does the Weighted average actuarial assumptions consist of? ",
#                      "context":"Actuarial assumptions The Group's scheme liabilities are measured using the projected unit credit method using the principal actuarial assumptions set out below: Notes: 1 Figures shown represent a weighted average assumption of the individual schemes. 2 The rate of increases in pensions in payment and deferred revaluation are dependent on the rate of inflation.|                        | 2019 % | 2018 % | 2017 % ||------------------------|--------|--------|--------|| Weighted average actuarial assumptions used at 31 March1: |          |        |        || Rate of inflation2     | 2.9    | 2.9    | 3.0    || Rate of increase in salaries | 2.7   | 2.7    | 2.6    || Discount rate          | 2.3    | 2.5    | 2.6    |"})  
# # text = format_input({"prompt": "Explain the concept of EBITDA and its significance in financial analysis."}),
# input_text = text
# inputs = tokenizer(input_text, return_tensors="pt")

# token_ids = generate_stream(
#     model=core_model,
#     idx=inputs['input_ids'].to(device),
#     max_new_tokens=500,
#     context_size=2048,  # Reduced context size to prevent memory issues
#     temperature=0.1,    # Lower temperature for more focused responses
#     top_k=3,            # More focused sampling
#     top_p=0.9,          # Nucleus sampling for quality
#     repetition_penalty=1.2,
#     eos_id=151643,
# )

# # Collect all tokens into a single response
# response = "".join([ i for i in token_ids])
# print(response)

@app.post("/predict/stream")
async def predict_stream(request: Request):
    """Streaming prediction endpoint using Server-Sent Events"""
    data = await request.json()
    text = data["prompt"]
    with open("prompts.log", "a") as f:
        f.write(f"Prompt: {text}\n")
    input_text = format_input({"prompt": text})
    inputs = tokenizer(input_text, return_tensors="pt")
    
    async def stream_generator():
        try:
            token_ids = generate_stream(
                model=core_model,
                idx=inputs['input_ids'].to(device),
                max_new_tokens=800,
                context_size=5012,
                eos_id=50256,
            )
    
            
            for token in token_ids:
                cleaned_token = token  # Fixed typo: was " ", not " "
                if cleaned_token:
                    yield cleaned_token
                await asyncio.sleep(0.01)
            yield "[DONE]"
        except Exception as e:
            yield f"ERROR: {str(e)}"
        finally:
            pass
            with open("predict.log", "a") as f:
                f.write(f"Prompt: {text}\n")

    return StreamingResponse(
        stream_generator(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash") 
def ask_gemini(query: str) -> str:
    global model
    prompt = query
    response = model.generate_content(prompt)
    json_pattern = r'\{.*\}'
    json_match = re.findall(json_pattern, response.text, flags=re.DOTALL)

    return json.loads(json_match[0])

@app.get("/suggestions")
async def suggestions():
    return JSONResponse(content = ask_gemini("I want you to suggest five potential questions I can ask an LLM that was trained purely on financial, SaaS, Social Media, Customer, and Customer Feedback analytics data. Return the questions in a JSON array in the format {\"questions\": [{\"metric_type\": \"Financial Analytics\", \"question\": \"question 1\"}, {\"metric_type\": \"SaaS\", \"question\": \"question 2\"}, {\"metric_type\": \"Social Media\", \"question\": \"question 3\"}, {\"metric_type\": \"Customer\", \"question\": \"question 4\"}, {\"metric_type\": \"Customer Feedback\", \"question\": \"question 5\"}]}"))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, access_log=False, log_level="info")