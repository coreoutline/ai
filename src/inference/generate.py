import torch


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.7,
    top_k=50,
    eos_id=None,
    repetition_penalty=1.2,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)["logits"]
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            penalize_len = min(idx_cond.shape[1], 100)
            for i in range(penalize_len):
                previous_token = idx_cond[0, -(i + 1)]
                logits[0, previous_token] = logits[0, previous_token] / repetition_penalty

        if top_k is not None:
            top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
