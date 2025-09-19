# app.py — FastAPI backend (chat + prédiction + / qui sert index.html)
# Lancer: uvicorn app:app --host 0.0.0.0 --port 8000
from typing import Dict, List, Optional
import re
import torch
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Unsloth prioritaire (chargement rapide), sinon Transformers
try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ========= Config =========
BASE_MODEL_ID = "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit"  # change si besoin

SYSTEM_CHAT = (
    "Tu es un assistant League of Legends. Réponds en français, clairement et utilement. "
    "Si l'utilisateur fournit deux compositions (Blue et Red) avec les rôles, "
    "tu peux analyser brièvement la draft si on te le demande."
)
SYSTEM_PRED = (
    "Tu es un analyste draft League of Legends. "
    "Quand deux compositions complètes (Top/Jungle/Mid/ADC/Support) sont fournies, "
    "donne : 1) Vainqueur (Blue/Red)  2) Confiance (faible/moyenne/haute)  "
    "3) 2–3 facteurs clés (prio lanes, engage, poke, scaling, anti-dive, etc.). "
    "Réponse courte."
)

LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{% if messages and messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% else %}You are a helpful assistant.{% endif %}<|eot_id|>
{% for message in messages if message['role'] != 'system' -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% endfor -%}
<|start_header_id|>assistant<|end_header_id|>
"""

ROLES = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]

# ========= Utils tokenizer / modèle =========
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
    out, seen = [], set()
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

def build_inputs(tok, messages: List[Dict[str, str]], device):
    ensure_chat_template(tok)
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids[0], list):
        input_ids = torch.tensor(ids, device=device)
    else:
        input_ids = torch.tensor([ids], device=device)
    attn = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attn}

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
            print("✅ Unsloth FastLanguageModel loaded")
            return model, tok
        except Exception as e:
            print(f"⚠️ Unsloth failed ({e}), fallback to Transformers.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    ensure_chat_template(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        low_cpu_mem_usage=True,
        offload_folder="offload",
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("✅ Transformers (4-bit) loaded")
    return model, tok

def generate(model, tok, messages: List[Dict[str, str]],
             max_new_tokens=512, temperature=0.7, top_p=0.9, min_new_tokens=0):
    device = next(model.parameters()).device
    batch = build_inputs(tok, messages, device)
    eos_ids = get_eos_ids(tok)

    gen_kwargs = dict(
        **batch,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=(eos_ids[0] if eos_ids else None),
        pad_token_id=tok.pad_token_id,
    )

    with torch.inference_mode():
        if "cuda" in str(device):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(**gen_kwargs)
        else:
            out = model.generate(**gen_kwargs)

    prompt_len = batch["input_ids"].shape[1]
    gen = out[0, prompt_len:]
    return tok.decode(gen, skip_special_tokens=True).strip()

# ========= Parsing compositions LoL =========
def parse_role_pairs(s: str) -> Dict[str, str]:
    comp = {r: "?" for r in ROLES}
    parts = [p.strip() for p in (s or "").split(",")]
    for p in parts:
        m = re.match(r"(?i)\s*(top|jungle|mid|adc|support)\s*=\s*(.+?)\s*$", p)
        if m:
            comp[m.group(1).upper()] = m.group(2)
    return comp

def comp_to_str(c: Dict[str, str]) -> str:
    return ", ".join(f"{r.title()}={c.get(r, '?')}" for r in ROLES)

def prediction_messages(blue: Dict[str, str], red: Dict[str, str]):
    blue_txt = comp_to_str(blue)
    red_txt  = comp_to_str(red)
    user = (
        "Prédire le vainqueur en **draft** :\n"
        f"- Blue : {blue_txt}\n"
        f"- Red  : {red_txt}\n\n"
        "Réponds par :\n"
        "1) Vainqueur (Blue/Red)\n"
        "2) Confiance (faible/moyenne/haute)\n"
        "3) 2–3 facteurs clés"
    )
    return [
        {"role": "system", "content": SYSTEM_PRED},
        {"role": "user", "content": user},
    ]

# ========= FastAPI =========
app = FastAPI(title="LoL Draft Helper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (facultatif) servir /static si présent
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Charger le modèle au démarrage
model, tok = load_base_model(BASE_MODEL_ID)

# ----- Schémas d'entrée -----
class ChatIn(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    max_new_tokens: Optional[int] = 512
    min_new_tokens: Optional[int] = 80
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    system: Optional[str] = None

class PredictIn(BaseModel):
    blue: str
    red: str
    max_new_tokens: Optional[int] = 512
    min_new_tokens: Optional[int] = 80

# ----- Routes -----
@app.get("/health")
def health():
    gpu = torch.cuda.is_available()
    return {
        "ok": True,
        "gpu": gpu,
        "device": (torch.cuda.get_device_name(0) if gpu else "CPU"),
        "model": BASE_MODEL_ID,
    }

@app.post("/api/chat")
def chat(inp: ChatIn):
    if inp.messages:
        msgs = inp.messages
        if not msgs or msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": inp.system or SYSTEM_CHAT}] + msgs
    else:
        if not inp.message:
            return {"text": "(message vide)"}
        msgs = [
            {"role": "system", "content": inp.system or SYSTEM_CHAT},
            {"role": "user", "content": inp.message.strip()},
        ]

    text = generate(
        model, tok, msgs,
        max_new_tokens=inp.max_new_tokens or 512,
        min_new_tokens=inp.min_new_tokens or 80,
        temperature=inp.temperature or 0.7,
        top_p=inp.top_p or 0.9,
    )
    return {"text": text}

@app.post("/api/predict")
def predict(inp: PredictIn):
    blue = parse_role_pairs(inp.blue)
    red = parse_role_pairs(inp.red)
    text = generate(
        model, tok, prediction_messages(blue, red),
        max_new_tokens=min(max(inp.max_new_tokens or 512, 64), 800),
        min_new_tokens=inp.min_new_tokens or 80,
        temperature=0.3,
        top_p=0.9,
    )
    return {"text": text}

@app.get("/")
def home():
    index_path = Path(__file__).with_name("index.html")
    return FileResponse(index_path if index_path.exists() else Path("index.html"))
