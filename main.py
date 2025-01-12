import logging
from constants import MAC_LOG_PATH, C2I_PATH, MODEL_NAME
from minlogscan import LogParser
from model import DraftPicker, prepare_inputs
import torch
import json
import time

logging.basicConfig(level=logging.INFO)

def handle_draft_start(parser):
    parser.search_draft_start()
    if parser.draft_start:
        return True
    return False


def handle_p1p1(parser):
    first_pick_done = parser.search_p1p1()
    return first_pick_done


def handle_pack_suggestion(parser, model, card_to_idx, device):
    # parser.search_pack()
    pack_names = parser.arenaid2names(parser.pack)
    pool_names = parser.arenaid2names(parser.pool)

    print("=== DRAFT UPDATE ===")
    print(f"Picked Pool ({len(pool_names)} cards): {pool_names}")
    print(f"Current Pack: {pack_names}")

    # Prepare inputs for the model
    pool_tensor, pack_tensor = prepare_inputs(pool_names, pack_names, card_to_idx, device)
    win_tensor = torch.tensor([7.0], dtype=torch.float32, device=device)
    rank_tensor = torch.tensor([6.0], dtype=torch.float32, device=device)

    # Get model predictions
    with torch.no_grad():
        logits = model(pool_tensor, pack_tensor, win_tensor, rank_tensor)
    raw_scores = {card: score.item() for card, score in zip(pack_names, logits.squeeze(0))}

    # Normalize scores (softmax)
    scores_tensor = torch.tensor(list(raw_scores.values()))
    normalized_scores_tensor = torch.nn.functional.softmax(scores_tensor, dim=0)
    normalized_scores = {k: v.item() for k, v in zip(raw_scores.keys(), normalized_scores_tensor)}

    # Choose the best card
    best_card = max(normalized_scores, key=normalized_scores.get)

    print("Card Scores:")
    for card, score in normalized_scores.items():
        print(f"  - {card}: {score:.2f}")
    print(f"Suggested Card: {best_card}")

    return best_card


def handle_user_pick(parser):
    if parser.search_pick():
        picked_card = parser.arena_id_map.get(parser.last_pick, "Unknown Card")
        logging.info(f"User picked: {picked_card}")
        print("====================\n")
        return True
    return False


if __name__ == "__main__":
    # Load resources
    with open(C2I_PATH, 'r') as f:
        card_to_idx = json.load(f)
    parser = LogParser(MAC_LOG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DraftPicker(embedding_dim=16, num_cards=len(card_to_idx)).to(device)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device, weights_only=True), strict=False)
    model.eval()

    # Flags and states
    draft_active = False
    wait_for_suggestion = False
    suggestion_printed = False
    last_processed_pack = [] # Track the last seen pack state

    logging.info("Starting draft loop.")
    while True:
        # Check if the draft has started
        if not draft_active:
            draft_active = handle_draft_start(parser)
            continue
        # parser.search_draft_data()

        # Handle Pack 1 Pick 1 (P1P1)
        if len(parser.pool) == 0 and not wait_for_suggestion:
            wait_for_suggestion = handle_p1p1(parser)
            continue

        # Handle subsequent packs
        if wait_for_suggestion:
            # Check if the current pack is new
            parser.search_pack()
            current_pack = parser.pack

            if current_pack != last_processed_pack:
                last_processed_pack = current_pack
                suggested_card = handle_pack_suggestion(parser, model, card_to_idx, device)
                suggestion_printed = True
            else:
                # Skip if it's the same pack
                continue

        # Check for user's pick
        if suggestion_printed:
            user_picked = handle_user_pick(parser)
            if user_picked:
                wait_for_suggestion = True  # Already True but for clarity
                suggestion_printed = False
            else:
                continue
        
        time.sleep(0.1)

