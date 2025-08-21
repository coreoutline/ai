from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import torch
from transformers import AutoTokenizer
from model_2 import core_model
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

core_model.load_state_dict(torch.load(f"./models/nyx_2.pth"))
core_model.eval()
core_model.to(device)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n ### Instruction:\n{entry['prompt'].strip().replace('<prompt>','').replace('</prompt>','')}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input'].strip()}" if "input" in entry else ""
    )
    return instruction_text + input_text + f"\n\n### Response: \n"

def generate_stream(model, idx, max_new_tokens, context_size, temperature=0.3, top_k=2, eos_id=None, repetition_penalty=1.2):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)['logits']
        logits = logits[:, -1, :]

        # Apply repetition penalty
        if repetition_penalty != 21.0:
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

        idx = torch.cat((idx, idx_next), dim=1)
        token = tokenizer.decode(idx_next[0])
        yield token

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
        max_new_tokens=300,
        context_size=100000,
        eos_id=151643,
    )
    
    # Collect all tokens into a single response
    response = "".join(token_ids)
    return {"response": response}

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
                max_new_tokens=100,
                context_size=10240,
                eos_id=151643,
            )
            
            for token in token_ids:
                cleaned_token = token.replace("<|endoftext|>", "")
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
    json_pattern = '\{.*\}'
    json_match = re.findall(json_pattern, response.text, flags=re.DOTALL)

    return json.loads(json_match[0])

@app.get("/suggestions")
async def suggestions():
    return JSONResponse(content = ask_gemini("I want you to suggest five potential questions I can ask an LLM that was trained purely on financial, SaaS, Social Media, Customer, and Customer Feedback analytics data. Return the questions in a JSON array in the format {\"questions\": [{\"metric_type\": \"Financial Analytics\", \"question\": \"question 1\"}, {\"metric_type\": \"SaaS\", \"question\": \"question 2\"}, {\"metric_type\": \"Social Media\", \"question\": \"question 3\"}, {\"metric_type\": \"Customer\", \"question\": \"question 4\"}, {\"metric_type\": \"Customer Feedback\", \"question\": \"question 5\"}]}"))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000, access_log=False, log_level="info")