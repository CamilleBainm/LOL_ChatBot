from datasets import load_dataset
from transformers import AutoTokenizer

# Charger ton dataset JSON
dataset = load_dataset("json", data_files="train.json", split="train")

# Charger le tokenizer LLaMA
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def format_chat(example):
    messages = [
        {"role": "system", "content": "Tu es un assistant expert en League of Legends."},
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]}
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": chat_text}

dataset = dataset.map(format_chat)
print(dataset[0]["text"])
