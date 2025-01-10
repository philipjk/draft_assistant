import csv
import os
from constants import CARDS_CSV_LINK
import requests
import logging

def load_arena_id_map(csv_path: str='cards.csv', expansion_code: str="FDN"):
    """Load a mapping {arena_id (int): card_name (str)} from a 17Lands CSV."""
    arena_id_map = {}
    if not os.path.exists(csv_path):
        logging.info(f"Downloading cards.csv from {CARDS_CSV_LINK}")
        response = requests.get(CARDS_CSV_LINK)
        with open(csv_path, 'wb') as f:
            f.write(response.content)
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            arena_id_str = row.get('id')  
            card_name = row.get('name')
            expansion = row.get('expansion')
            
            if arena_id_str and card_name and expansion == expansion_code:
                try:
                    arena_id = int(arena_id_str)
                    arena_id_map[arena_id] = card_name
                except ValueError:
                    pass
    return arena_id_map