import os
import json
import torch
from unsloth import FastLanguageModel

# -------------------------------
# Trouver le plus gros checkpoint
# -------------------------------
output_dir = "outputs"
checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
if not checkpoints:
    raise ValueError("Aucun checkpoint trouvé dans outputs/ !")

# trie par numéro de step
latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
checkpoint_path = os.path.join(output_dir, latest_checkpoint)
print(f"🔄 Chargement du checkpoint: {checkpoint_path}")

# -------------------------------
# Charger modèle et tokenizer
# -------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    max_seq_length=2048,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=True,
    device_map="auto",
)

# -------------------------------
# Charger dataset de test
# -------------------------------
with open("test.json", "r", encoding="utf-8") as f:   # change si ton fichier diffère
    test_data = json.load(f)

# -------------------------------
# Fonction pour interroger le modèle
# -------------------------------
def ask_model(user_input):
    messages = [
        {"role": "system", "content": "Tu es un analyste qui prédit l’issue d’un match de League of Legends. Réponds uniquement par 'Victoire de l'équipe bleue' ou 'Victoire de l'équipe rouge'."},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,      # réponse courte
            do_sample=True,         # sampling activé
            temperature=0.8,        # plus de diversité
            top_p=0.9,              # nucleus sampling
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# -------------------------------
# Évaluation
# -------------------------------
correct = 0
total = len(test_data)

for example in test_data:
    expected = example["output"].strip().lower()
    user_input = example["instruction"] + "\n\n" + example["input"]

    response = ask_model(user_input).lower()

    if expected in response:
        correct += 1

    print(f"Prompt: {user_input}")
    print(f"Attendu: {expected}")
    print(f"Réponse IA: {response}")
    print("✅ Correct\n" if expected in response else "❌ Incorrect\n")

accuracy = 100 * correct / total
print(f"\n📊 Accuracy: {accuracy:.2f}% ({correct}/{total})")
