import pandas as pd
import json
 
df = pd.read_csv("games_test.csv")
 
records = []
for _, row in df.iterrows():
    # récupère les picks (tes colonnes doivent correspondre aux rôles)
    blue_team = f"Top={row['blue_top']}, Jungle={row['blue_jungle']}, Mid={row['blue_mid']}, ADC={row['blue_adc']}, Support={row['blue_support']}"
    red_team  = f"Top={row['red_top']}, Jungle={row['red_jungle']}, Mid={row['red_mid']}, ADC={row['red_adc']}, Support={row['red_support']}"
 
    winner = "Victoire de l'équipe bleue" if row["winner"] == 100 else "Victoire de l'équipe rouge"
 
    record = {
        "instruction": "Prédire quelle équipe gagne cette partie de League of Legends.",
        "input": f"Blue: {blue_team}\nRed: {red_team}",
        "output": winner
    }
    records.append(record)
 
with open("test.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
 
 