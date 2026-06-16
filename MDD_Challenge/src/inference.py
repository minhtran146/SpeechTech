import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from config import Config
from data import MDDDataset, DataCollatorCTCWithPadding, load_vocab
from model import create_model


@torch.no_grad()
def predict(model, dataloader, device, blank_id=0):
    model.eval()
    all_preds = []

    for batch in dataloader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        canonical_ids = batch["canonical_ids"].to(device)

        output = model(input_values, canonical_ids, attention_mask=attention_mask)
        logits = output.logits
        pred_ids = torch.argmax(logits, dim=-1)

        for b in range(pred_ids.shape[0]):
            ids = pred_ids[b].tolist()
            collapsed = []
            prev = None
            for tid in ids:
                if tid != prev and tid != blank_id:
                    collapsed.append(tid)
                prev = tid
            all_preds.append(collapsed)

    return all_preds


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for public test set")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to public test CSV (e.g., MDD-Challenge-2025-public-test/metadata/public_test_phones.csv)")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Path to audio directory (e.g., MDD-Challenge-2025-public-test/audio_data)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default="outputs/vocab.json",
                        help="Path to vocabulary file")
    parser.add_argument("--processor", type=str, default="outputs/processor",
                        help="Path to processor directory")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.test_csv)
    
    # --- ĐOẠN MÃ THÊM VÀO ĐỂ XỬ LÝ PRIVATE TEST ---
    # Bơm dữ liệu giả cho DataLoader nếu các cột này không tồn tại
    if "transcript" not in df.columns:
        df["transcript"] = "[CTC_BLANK]"
    if "canonical" not in df.columns:
        df["canonical"] = "[CTC_BLANK]"
    # -----------------------------------------------

    print(f"Test samples: {len(df)}")

    vocab = load_vocab(args.vocab)
    vocab_size = len(vocab)
    id2token = {v: k for k, v in vocab.items()}
    print(f"Vocab size: {vocab_size}")

    cfg = Config()
    model = create_model(cfg, vocab_size)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.to(device)
    print("Model loaded successfully")

    processor = Wav2Vec2Processor.from_pretrained(args.processor)

    test_dataset = MDDDataset(df, args.audio_dir, vocab, cfg)
    collator = DataCollatorCTCWithPadding(processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    predictions = predict(model, test_loader, device)

    results = []
    for pred_ids in predictions:
        tokens = [id2token.get(tid, "") for tid in pred_ids]
        tokens = [t for t in tokens if t and t != "[CTC_BLANK]"]
        results.append(" ".join(tokens))

    results_df = pd.DataFrame({
        "id": df["id"],
        "predict": results
    })
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Sample predictions:")
    print(results_df.head(5))


if __name__ == "__main__":
    main()