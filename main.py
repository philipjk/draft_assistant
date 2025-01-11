import logging
from constants import MAC_LOG_PATH, C2I_PATH, MODEL_NAME
from minlogscan import LogParser
from model import DraftPicker, prepare_inputs
import torch
import json

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    with open(C2I_PATH, 'r') as f:
        card_to_idx = json.load(f)
    parser = LogParser(MAC_LOG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DraftPicker(embedding_dim=16, num_cards=len(card_to_idx)).to(device)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device,weights_only=True),
                          strict=False)
    model.to(device)
    model.eval()

    last_pool_size = 0
    while True:
        parser.search_draft_start()
        if parser.draft_start:
            if len(parser.pool) == 0:
                # no pack data for pack 1 pick 1
                parser.search_p1p1()
            else:
                parser.search_pack()
                pack_names = parser.arenaid2names(parser.pack)
                pool_names = parser.arenaid2names(parser.pool)
                print("=== DRAFT UPDATE ===")
                print(f"Picked Pool ({len(pool_names)} cards): {pool_names}")
                print(f"Current Pack: {pack_names}")
                pool_tensor, pack_tensor = prepare_inputs(pool_names, pack_names, card_to_idx, device)
                win_tensor = torch.tensor([6.0], dtype=torch.float32, device=device)
                rank_tensor = torch.tensor([6.0], dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits = model(pool_tensor, pack_tensor, win_tensor, rank_tensor)
                raw_scores = {card: score.item() for card, score in zip(pack_names, logits.squeeze(0))} 
                # Normalize scores (softmax)
                scores_tensor = torch.tensor(list(raw_scores.values()))
                normalized_scores_tensor = torch.nn.functional.softmax(scores_tensor, dim=0)
                normalized_scores = {
                    k: v.item()
                    for k, v in zip(raw_scores.keys(), normalized_scores_tensor)
                }

                # Choose the best card
                best_card = max(normalized_scores, key=normalized_scores.get)

                print("Card Scores:")
                for card, score in normalized_scores.items():
                    print(f"  - {card}: {score:.2f}")
                print(f"Suggested Card: {best_card}")
                parser.search_pick()
                print(f"Picked Card: {parser.arena_id_map.get(parser.last_pick, 'Unknown Card')}")
                print("====================\n")
                if len(parser.pool) == len(pool_names):
                    break
            last_pool_size = len(parser.pool)
