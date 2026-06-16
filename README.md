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
└── outputs/              # Checkpoints, vocab, predictions
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Train model (local)

```bash
cd MDD_Challenge
python -m src.main
```

### 2. Train model (Google Colab)

Mở `mdd-train.ipynb` trong Colab, chạy lần lượt các cell:

| Cell | Mô tả |
|------|-------|
| 1 | Mount Google Drive |
| 2 | `cd` vào thư mục dự án |
| 3 | Cài đặt dependencies |
| 4 | Train model (`python src/main.py`) |
| 5 | Inference public test |
| 6 | Evaluation |
| 7 | Inference private test |

Đảm bảo dữ liệu đã được upload lên Google Drive đúng cấu trúc.

### 3. Inference (test set)

```bash
python -m src.inference \
    --test_csv path/to/test_phones.csv \
    --audio_dir path/to/audio_data \
    --output predictions.csv
```

### 4. Evaluation

```bash
python MDD-Metrics/evaluate.py ground_truth.csv results.csv
```

## Cấu trúc dữ liệu

Dữ liệu cần có cấu trúc như sau (có thể copy từ challenge):

```
MDD-Challenge-2025-training-set/
├── metadata/
│   ├── train.csv              # Câu readable
│   ├── train_phones.csv       # Câu dạng âm vị
│   └── lexicon_vmd.txt        # Từ điển phát âm
└── audio_data/
    └── train/                 # File .wav
        ├── sample1.wav
        └── ...
```

### Thêm dữ liệu mới

1. Thêm file `.wav` vào `audio_data/train/`
2. Thêm dòng tương ứng vào `metadata/train_phones.csv` với các cột:
   - `id`: định danh mẫu
   - `path`: đường dẫn file `.wav`
   - `canonical`: phát âm chuẩn (dạng âm vị, cách nhau bởi space, từ cách bởi `$`)
   - `transcript`: phát âm thực tế (dạng âm vị)
3. Cập nhật `data_root` trong `src/config.py` nếu thư mục khác tên
4. Chạy lại `python -m src.main`

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
