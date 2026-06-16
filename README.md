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

Dự án MDD Challenge có 3 bộ dữ liệu riêng biệt, mỗi bộ nằm trong thư mục gốc riêng:

```
Thư mục dự án/
├── MDD-Challenge-2025-training-set/         # Dữ liệu train (3180 file wav)
│   ├── metadata/
│   │   ├── train.csv                        # Câu readable (id, path, canonical, transcript)
│   │   ├── train_phones.csv                 # Câu dạng âm vị → dùng để train
│   │   └── lexicon_vmd.txt                  # Từ điển phát âm (word → phonemes)
│   └── audio_data/
│       └── train/                           # File .wav cho train
│           ├── F_F_0001.wav
│           ├── F_F_0002.wav
│           └── ...
│
├── MDD-Challenge-2025-public-test/          # Dữ liệu public test (dùng để validation)
│   ├── metadata/
│   │   └── public_test_phones.csv           # Không có cột transcript
│   └── audio_data/
│       └── public_test/
│           ├── F_F_1001.wav
│           └── ...
│
└── MDD-Challenge-2025-private-test/         # Dữ liệu private test (dùng để submission)
    ├── metadata/
    │   └── private_test_submission.csv      # File mẫu để điền kết quả
    └── audio_data/
        └── private_test/
            ├── F_F_2001.wav
            └── ...
```

### Cách cấu hình đường dẫn trong code

| Biến trong `config.py` | Giá trị mặc định | Mục đích |
|------------------------|-----------------|----------|
| `data_root` | `MDD-Challenge-2025-training-set` | Thư mục gốc chứa dữ liệu train |
| `metadata_path` | `{data_root}/metadata/train_phones.csv` | File CSV âm vị dùng để train |
| `audio_dir` | `{data_root}/audio_data/train` | Thư mục chứa file `.wav` train |

Nếu bạn đổi tên thư mục hoặc cấu trúc, chỉ cần sửa `data_root` trong `src/config.py`.

### Thêm dữ liệu mới

1. Thêm file `.wav` vào `audio_data/train/`
2. Thêm dòng tương ứng vào `metadata/train_phones.csv` với format:
   ```csv
   id,path,canonical,transcript
   new_001,audio_data/train/new_001.wav,ɓ aː-4 k ɔ-4 $ m ɛ-3,ɓ aː-4 t ɔ-4 $ m ɛ-3
   ```
   - `id`: định danh duy nhất
   - `path`: đường dẫn file `.wav`
   - `canonical`: phát âm chuẩn (dạng âm vị, cách nhau bởi space, từ cách bởi `$`)
   - `transcript`: phát âm thực tế (dạng âm vị) — có thể giống canonical nếu đọc đúng
3. Chạy lại `python -m src.main`

### Chạy inference trên public/private test

Xem hướng dẫn trong notebook `mdd-train.ipynb` (cells 5-7) hoặc chạy trực tiếp:

```bash
# Public test
python -m src.inference \
    --test_csv MDD-Challenge-2025-public-test/metadata/public_test_phones.csv \
    --audio_dir MDD-Challenge-2025-public-test/audio_data/public_test \
    --output predictions_public.csv

# Private test
python -m src.inference \
    --test_csv MDD-Challenge-2025-private-test/metadata/private_test_submission.csv \
    --audio_dir MDD-Challenge-2025-private-test/audio_data/private_test \
    --output predictions_private.csv
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
