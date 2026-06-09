import torch
import tiktoken
from model import CoreModel
import time
from transformers import AutoTokenizer
# import torch.quantization as quant
# import openvino as ov


from model_2 import core_model
import requests
import os

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



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


num_workers = 4
batch_size = 8

torch.manual_seed(123)




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
        # if idx_next == eos_id:
        #     break
        idx = torch.cat((idx, idx_next), dim=1)
        # response = (tokenizer.decode(idx_next[0]))
        # print(response)
    return idx


def predict(text):

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_text = format_input({"prompt":text})
    print(input_text)
    inputs = tokenizer(input_text, return_tensors="pt")

    
    core_model.load_state_dict(torch.load(f"./models/nyx_2.pth"))
    core_model.eval()
    try:
        core_model.to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"CUDA error: {e}\nFalling back to CPU.")
        device_cpu = torch.device("cpu")
        core_model.to(device_cpu)
        device = device_cpu

    # example = torch.randn(1, 3, 224, 224)
    # ov_model = ov.convert_model(core_model, example_input=(example,))
    # core = ov.Core()
    # compiled_model = core.compile_model(ov_model, 'CPU')



    # quantized_model = quant.quantize_dynamic(
    #     core_model, {torch.nn.Linear}, dtype=torch.qint8
    # )
    # torch.save(core_model, f"/root/transformer/nyx_model.pth")
    # torch.onnx.export(
    #     core_model,
    #     inputs['input_ids'],                          # model input
    #     "nyx.onnx",                       # where to save the ONNX model
    #     input_names=["input_ids"],
    #     output_names=["logits"],
    #     dynamic_axes={                       # makes it flexible for batch size and seq length
    #         "input_ids": {0: "batch_size", 1: "sequence_length"},
    #         "logits": {0: "batch_size", 1: "sequence_length"}
    #     },
    #     opset_version=13
    # )

    # quantized_model.eval()
    # quantized_model.to("cpu")
    torch.manual_seed(123)

    
    print(inputs)
    token_ids = generate(
        model=core_model,
        idx=inputs['input_ids'].to(device),
        max_new_tokens=100,
        context_size=5012,
        eos_id=151643,
    )

    response = (tokenizer.decode(token_ids[0]))
    print(response)

    end_time = time.time()
    print(f"Prediction took {(end_time - start_time) / 60:.2f} minutes")
    return response



predict("What is your name?")