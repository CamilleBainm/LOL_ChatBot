# test.py — Chat interactif SANS checkpoint (base model only), robuste sur Unsloth/HF
# Usage :
#   python test.py
#   python test.py --base_model unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit

import argparse
from typing import List, Dict

import torch

try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Chat template Llama 3 (fallback si le tokenizer n’en a pas)
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{% if messages and messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% else %}You are a helpful assistant.{% endif %}<|eot_id|>
{% for message in messages if message['role'] != 'system' -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% endfor -%}
<|start_header_id|>assistant<|end_header_id|>
"""

def ensure_chat_template(tok):
    if not getattr(tok, "chat_template", None):
        tok.chat_template = LLAMA3_CHAT_TEMPLATE

def get_eos_ids(tok):
    ids = []
    if getattr(tok, "eos_token_id", None) is not None:
        ids.append(tok.eos_token_id)
    try:
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            ids.append(eot)
    except Exception:
        pass
    # dédoublonne
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

def load_base_model(model_id: str):
    print(f"🔹 Loading base model: {model_id}")
    if FastLanguageModel is not None:
        try:
            model, tok = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=4096,
                dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
                load_in_4bit=True,
                device_map="auto",
            )
            ensure_chat_template(tok)
            print("✅ Loaded with Unsloth FastLanguageModel")
            return model, tok
        except Exception as e:
            print(f"⚠️ Unsloth load failed ({e}). Falling back to Transformers...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    ensure_chat_template(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        offload_folder="offload",
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("✅ Loaded with Transformers (4-bit)")
    return model, tok

def build_inputs(tok, messages: List[Dict[str,str]], device):
    """
    🔒 Robuste : on demande directement les IDs au template
    -> pas de `tokenizer(prompt, ...)` intermédiaire
    """
    ensure_chat_template(tok)
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids[0], list):  # certains tokenizers renvoient déjà batché
        input_ids = torch.tensor(ids, device=device)
    else:
        input_ids = torch.tensor([ids], device=device)
    attn = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attn}

def generate(model, tok, messages: List[Dict[str,str]],
             max_new_tokens=256, temperature=0.7, top_p=0.9):
    device = next(model.parameters()).device
    batch = build_inputs(tok, messages, device)
    eos_ids = get_eos_ids(tok)

    with torch.inference_mode():
        if "cuda" in str(device):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=(eos_ids[0] if eos_ids else None),
                    pad_token_id=tok.pad_token_id,
                )
        else:
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=(eos_ids[0] if eos_ids else None),
                pad_token_id=tok.pad_token_id,
            )

    prompt_len = batch["input_ids"].shape[1]
    gen = out[0, prompt_len:]
    return tok.decode(gen, skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser(description="Chat")
    ap.add_argument("--base_model", type=str,
                    default="unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
                    help="HF model id (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct)")
    ap.add_argument("--system", type=str,
                    default="Tu es un expert en League of Legends. Donne des conseils précis.",
                    help="System prompt")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    model, tok = load_base_model(args.base_model)
    device = next(model.parameters()).device
    print(f"💻 Device: {device} — CUDA dispo: {torch.cuda.is_available()}")
    print("🤖 Base model ready! Type 'quit' to exit.\n")

    history: List[Dict[str,str]] = [{"role":"system","content": args.system}]

    while True:
        try:
            user = input("🧑 Toi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in {"quit","exit","stop"}:
            print("Bye.")
            break

        history.append({"role":"user","content": user})
        reply = generate(
            model, tok, history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        history.append({"role":"assistant","content": reply})
        print(f"🤖 IA: {reply}\n")

if __name__ == "__main__":
    main()
