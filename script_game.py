import os
import time
import requests
import pandas as pd
import urllib.parse
import random

# -----------------------------
# Config
# -----------------------------
API_KEY = os.environ["RIOT_API_KEY"]
HEADERS = {"X-Riot-Token": API_KEY}

TIER = "PLATINUM"
DIVISION = "I"
PATCH = "15.3"
QUEUE_ID = 420
ROLE_LIST = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
MAX_MATCHES_PER_PLAYER = 20
NB_JOUEURS = 4000

PLATFORM = "euw1"
REGION = "europe"

OUTPUT_FILE = "games_test.csv"

# -----------------------------
# Helpers pour API
# -----------------------------
def safe_get(url, params=None, max_retries=6):
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.ReadTimeout:
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            retry = int(r.headers.get("Retry-After", 1))
            time.sleep(retry)
            backoff *= 2
        elif 500 <= r.status_code < 600:
            time.sleep(backoff)
            backoff *= 2
        else:
            return None
    return None

# -----------------------------
# API calls
# -----------------------------
def get_league_entries(tier=TIER, division=DIVISION, queue="RANKED_SOLO_5x5", page=1):
    url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/entries/{queue}/{tier}/{division}"
    return safe_get(url, params={"page": page})

def get_match_ids(puuid, count=MAX_MATCHES_PER_PLAYER):
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    return safe_get(url, params={"queue": QUEUE_ID, "start":0, "count":count}) or []

def get_match(match_id):
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return safe_get(url)

def get_champion_mapping():
    url = "https://ddragon.leagueoflegends.com/cdn/15.3.1/data/fr_FR/champion.json"
    data = requests.get(url).json()
    champ_map = {}
    for champ in data["data"].values():
        champ_map[int(champ["key"])] = champ["name"]
    return champ_map

# -----------------------------
# Extraction match → ligne CSV
# -----------------------------
def extract_match_row(match, champ_map):
    if not match or "info" not in match:
        return None
    info = match["info"]

    # Ignore les remakes ou matchs trop courts
    if info.get("gameDuration", 0) < 300:
        return None

    # Vérifie le patch
    game_patch = ".".join(info.get("gameVersion", "").split(".")[:2])
    if not game_patch.startswith("15."):
        return None

    # Récupère les participants filtrés par rôle
    participants = info.get("participants", [])
    blue = [p for p in participants if p["teamId"] == 100 and p.get("teamPosition") in ROLE_LIST] 
    red  = [p for p in participants if p["teamId"] == 200 and p.get("teamPosition") in ROLE_LIST]

    # Vérifie que chaque équipe est complète
    if len(blue) != 5 or len(red) != 5:
        return None

    # Détermine le vainqueur
    if all(p["win"] for p in blue):
        winner = 100
    elif all(not p["win"] for p in blue):
        winner = 200
    else:
        winner = 0  # match nul ou résultat incohérent

    return {
        "patch": game_patch,
        "duration": info.get("gameDuration", 0),
        "blue_top":     champ_map.get(next(p["championId"] for p in blue if p["teamPosition"]=="TOP")),
        "blue_jungle":  champ_map.get(next(p["championId"] for p in blue if p["teamPosition"]=="JUNGLE")),
        "blue_mid":     champ_map.get(next(p["championId"] for p in blue if p["teamPosition"]=="MIDDLE")),
        "blue_adc":     champ_map.get(next(p["championId"] for p in blue if p["teamPosition"]=="BOTTOM")),
        "blue_support": champ_map.get(next(p["championId"] for p in blue if p["teamPosition"]=="UTILITY")),
        "red_top":      champ_map.get(next(p["championId"] for p in red if p["teamPosition"]=="TOP")),
        "red_jungle":   champ_map.get(next(p["championId"] for p in red if p["teamPosition"]=="JUNGLE")),
        "red_mid":      champ_map.get(next(p["championId"] for p in red if p["teamPosition"]=="MIDDLE")),
        "red_adc":      champ_map.get(next(p["championId"] for p in red if p["teamPosition"]=="BOTTOM")),
        "red_support":  champ_map.get(next(p["championId"] for p in red if p["teamPosition"]=="UTILITY")),
        "winner": winner
    }




# -----------------------------
# Sauvegarde incrémentale
# -----------------------------
def append_row_to_csv(row, filename=OUTPUT_FILE):
    df = pd.DataFrame([row])
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode="a", index=False, header=False)

# -----------------------------
# Pipeline principal
# -----------------------------
def main():
    all_entries = []
    max_pages = 50
    for page in range(1, max_pages + 1):
        entries = get_league_entries(page=page)
        if not entries:
            break
        all_entries.extend(entries)
    print(f"Nombre total de joueurs récupérés: {len(all_entries)}")
    if not all_entries:
        return

    random.shuffle(all_entries)
    all_entries = all_entries[:NB_JOUEURS]
    puuids = [e["puuid"] for e in all_entries if "puuid" in e]
    champ_map = get_champion_mapping()

    for puuid in puuids:
        ids = get_match_ids(puuid)
        for mid in ids:
            match = get_match(mid)
            if not match:
                continue
            row = extract_match_row(match, champ_map)
            if row:
                append_row_to_csv(row)
                print(f"[SAVE] Match sauvegardé (patch {row['patch']}, winner {row['winner']}).")


if __name__ == "__main__":
    main()
