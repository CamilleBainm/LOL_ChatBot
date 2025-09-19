import os
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# -------------------------------
# Toujours repartir de zéro
# -------------------------------
output_dir = "outputs"

# -------------------------------
# Charger modèle et tokenizer
# -------------------------------
model_name = "unsloth/llama-3.1-8b-instruct"   # si OOM -> remplace par "unsloth/llama-3.1-3b-instruct"
max_seq_length = 2048
torch.backends.cuda.matmul.allow_tf32 = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=True,
    device_map="auto",
)

# -------------------------------
# Charger dataset local
# -------------------------------
file = json.load(open("train.json", "r"))

def format_chat(example):
    messages = [
        {"role": "system", "content": "Tu es un assistant expert en League of Legends."},
        {"role": "user", "content": example["instruction"] + "\n\n" + example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

formatted_data = [format_chat(item) for item in file]
dataset = Dataset.from_dict({"text": formatted_data})

# -------------------------------
# Ajouter LoRA adapters
# -------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# -------------------------------
# Entraînement (toujours fresh start)
# -------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=1,   # limite VRAM
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        dataloader_pin_memory=False,
        report_to="none",
    ),
)

trainer.train()

# -------------------------------
# Sauvegarde finale du modèle fine-tuné
# -------------------------------
final_dir = os.path.join(output_dir, "finetuned_lora")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"✅ Modèle fine-tuné sauvegardé dans {final_dir}")
