# Chest X-Ray Pneumonia CNN (Assignment 1, Q1)

Dataset: [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Mooney), binary NORMAL/PNEUMONIA.
Framework: PyTorch. Seed: `42` everywhere (data split, subsampling, grid search, training).

The dataset's own train/val/test split is unbalanced (val has 16 images) and is not used.
`src/data_prep.py` pools all images and re-splits 80/10/10 (stratified, seed 42).

## Local development (no GPU, no real data)

All logic lives in `src/` and is covered by `pytest` against synthetic tiny images/manifests —
no dataset download required to verify correctness:

```bash
pip install torch torchvision pillow pytest  # scikit-learn/matplotlib not required; see note below
python3 -m pytest tests/ -q
```

Note: confusion-matrix/ROC plotting (`evaluate.py`, `error_analysis.py`) uses `matplotlib` if
installed, and silently skips plotting otherwise, since Kaggle notebooks ship it preinstalled.

## Running on Kaggle (real GPU training)

1. Add the dataset via Kaggle's "Add Data" (`paultimothymooney/chest-xray-pneumonia`), mounted at
   `/kaggle/input/chest-xray-pneumonia`.
2. Upload/clone this repo's `src/` into the notebook working directory.
3. Run each stage via shell-out cells (`!python3 -m src.<module> --flags`):

```bash
# 1. Build stratified 80/10/10 manifests (dedupes, drops corrupt files)
python3 -m src.data_prep \
  --data-dir /kaggle/input/chest-xray-pneumonia/chest_xray \
  --out-dir /kaggle/working/manifests --seed 42

# 2. Train the custom CNN (also supports --model resnet50|mobilenetv2, --mode frozen|finetune)
python3 -m src.train \
  --model cnn --mode scratch \
  --train-manifest /kaggle/working/manifests/train.csv \
  --val-manifest /kaggle/working/manifests/val.csv \
  --epochs 30 --checkpoint-dir /kaggle/working/checkpoints \
  --log-path /kaggle/working/logs/cnn_scratch.csv

# 3. Grid search (random search over the full space; writes results.csv + summary.csv)
python3 -m src.tune \
  --model cnn --mode scratch \
  --train-manifest /kaggle/working/manifests/train.csv \
  --val-manifest /kaggle/working/manifests/val.csv \
  --n-trials 20 --epochs-per-trial 10 --seed 42 \
  --out-dir /kaggle/working/tune

# 4. Data-size study (25/50/75/100% of train)
python3 -m src.data_size_study \
  --model cnn --mode scratch \
  --train-manifest /kaggle/working/manifests/train.csv \
  --val-manifest /kaggle/working/manifests/val.csv \
  --fractions 0.25,0.5,0.75,1.0 --epochs 30 --seed 42 \
  --out-dir /kaggle/working/data_size_study

# 5. Evaluate a trained checkpoint (appends a row to the cross-model comparison table)
python3 -m src.evaluate \
  --model cnn --mode scratch \
  --checkpoint /kaggle/working/checkpoints/cnn_scratch_best.pth \
  --test-manifest /kaggle/working/manifests/test.csv \
  --out-dir /kaggle/working/eval/cnn_scratch \
  --comparison-csv /kaggle/working/eval/comparison.csv

# 6. Error analysis (top-10 misclassified samples + grid image)
python3 -m src.error_analysis \
  --model cnn --mode scratch \
  --checkpoint /kaggle/working/checkpoints/cnn_scratch_best.pth \
  --test-manifest /kaggle/working/manifests/test.csv \
  --out-dir /kaggle/working/errors/cnn_scratch
```

Repeat steps 2/5/6 for `--model resnet50 --mode frozen`, `--model resnet50 --mode finetune`,
`--model mobilenetv2 --mode frozen`, `--model mobilenetv2 --mode finetune` to fill out the
comparative table required by the report (Task 8).

## Layout

```
src/
  data_prep.py         # discover/dedupe/filter/split/subsample -> manifest CSVs
  dataset.py            # ChestXrayDataset, augmentation, class-imbalance sample weights
  models/cnn.py          # custom CNN + layer_table() for the report
  models/pretrained.py   # ResNet50 / MobileNetV2 frozen & fine-tune wrappers
  train.py               # training loop, early stopping, checkpointing, CLI
  tune.py                # random search over the hyperparameter grid, CLI
  data_size_study.py     # 25/50/75/100% data-fraction sweep, CLI
  evaluate.py            # metrics, AUC-ROC, confusion/ROC plots, comparison table, CLI
  error_analysis.py      # misclassified sample extraction + grid, CLI
tests/                   # pytest, synthetic data only
```

Git history: one branch per task (`task1-data-prep` ... `task9-error-analysis`), each following
strict TDD (failing test committed, then the minimal implementation), merged into `main`.
