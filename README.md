# SpeechTech

Mispronunciation Detection & Diagnosis (MDD) for Vietnamese speech.

Dự án sử dụng **wav2vec2 + CTC fine-tuning** để phát hiện và chẩn đoán lỗi phát âm tiếng Việt từ file `.wav`.

## Cấu trúc thư mục

```
MDD_Challenge/
├── src/                  # Mã nguồn Python (modular, tái sử dụng)
│   ├── config.py         # Cấu hình (dataclass)
│   ├── data.py           # Dataset, DataLoader, Vocabulary, Processor
│   ├── model.py          # PLModel (Wav2Vec2 + Phonetic/Linguistic Encoder)
│   ├── train.py          # Training loop (với AMP, Gradual Unfreeze, SWA)
│   ├── inference.py      # Inference cho test set
│   ├── eval.py           # Evaluation với MDD-Metrics
│   └── main.py           # Entry point (train + eval)
├── MDD-Metrics/          # Official evaluation script (F1, DER, PER)
├── mdd-train.ipynb       # Colab notebook
├── outputs/              # Checkpoints, vocab, predictions
├── requirements.txt
└── AGENTS.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Train model

```bash
cd MDD_Challenge
python -m src.main
```

### 2. Inference (test set)

```bash
python -m src.inference \
    --test_csv path/to/test_phones.csv \
    --audio_dir path/to/audio_data \
    --output predictions.csv
```

### 3. Evaluation

```bash
python MDD-Metrics/evaluate.py ground_truth.csv results.csv
```

## Kiến trúc model

```
Audio ─→ Wav2Vec2 ─→ PhoneticEncoder ─→ CrossAttention ─→ GatedFusion ─→ LM Head → CTC Loss
                        ↑                                            ↑
Canonical ─→ LinguisticEncoder ──────────────────────────────────────┘
```

## Tính năng chính

- **Modular design**: `config.py`, `data.py`, `model.py`, `train.py`, `inference.py` độc lập
- **Mixed precision (AMP)**: Training nhanh hơn với `float16`
- **Data augmentation**: Time-stretch + Gaussian noise
- **Gradual Unfreeze**: Mở CNN sau epoch 25 với LR nhỏ
- **SWA**: Stochastic Weight Averaging top-3 checkpoints
- **CTC Loss** với `zero_infinity=True` tránh NaN

## Cấu hình

Xem và sửa `src/config.py` để điều chỉnh:
- `model_name`: pretrained wav2vec2 backbone
- `num_epochs`: số epoch (mặc định 50)
- `unfreeze_at_epoch`: epoch mở CNN (mặc định 25)
- `batch_size`, `learning_rate`, `dropout`, ...
