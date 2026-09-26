#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen3_server.py
───────────────
Lightweight OpenAI-compatible API server for Qwen3-8B.
Loads the model with device_map="auto" so weights are spread across
all available GPUs (e.g. 4x3090), avoiding single-card OOM.

Exposes:
    POST /v1/chat/completions   — standard OpenAI chat format
    GET  /v1/models             — model list (for connectivity check)

Usage
-----
    conda activate vllm_env
    python qwen3_server.py

    # Custom port or model:
    python qwen3_server.py --port 8001 \
        --model /workspace/SAGE-3D_Official/Qwen/Qwen/Qwen3-8B

Compatible with any OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    resp = client.chat.completions.create(model="qwen3", messages=[...])
"""
from __future__ import annotations

import argparse
import re
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=
                   "/workspace/SAGE-3D_Official/Qwen/Qwen/Qwen3-8B")
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    return p.parse_args()

ARGS = parse_args()

# ── Load model once at startup ─────────────────────────────────────────────
print(f"[SERVER] Loading {ARGS.model} with device_map='auto' ...")
tokenizer = AutoTokenizer.from_pretrained(ARGS.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ARGS.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",          # spreads across all GPUs automatically
    trust_remote_code=True,
)
model.eval()
print(f"[SERVER] Model loaded. Device map: {model.hf_device_map}")

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen3 OpenAI-compatible server")


# ── Request / Response schemas ─────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen3"
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 512
    stream: bool = False


# ── /v1/models ────────────────────────────────────────────────────────────
@app.get("/v1/models")
def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "qwen3",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    })


# ── /v1/chat/completions ──────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # Append /no_think to last user message to suppress thinking tokens
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            if "/no_think" not in messages[i]["content"]:
                messages[i] = {**messages[i],
                               "content": messages[i]["content"] + " /no_think"}
            break

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    do_sample = req.temperature > 0.0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=min(req.max_tokens, ARGS.max_new_tokens),
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = req.temperature

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0

    new_tokens = output_ids[0][input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Strip any residual <think>...</think> blocks
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

    completion_tokens = len(new_tokens)
    print(f"[SERVER] {completion_tokens} tokens in {elapsed:.1f}s "
          f"({completion_tokens/elapsed:.1f} tok/s)")

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": raw},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     input_len,
            "completion_tokens": completion_tokens,
            "total_tokens":      input_len + completion_tokens,
        },
    })


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[SERVER] Starting on http://{ARGS.host}:{ARGS.port}")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")