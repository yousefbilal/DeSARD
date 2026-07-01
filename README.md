# Training Scripts

This repository includes two simple training scripts (along with the sythetic data generation script with its own README file):

- `train_classifier.py` trains an image classifier for `human` vs `without human`.
- `train_yolo.py` trains a YOLO detector using the YOLO labels.

Both scripts use only `dataset`.

## Install

Install dependencies with CUDA 12.8 PyTorch wheels:

```powershell
pip install -r requirements.txt
```

## Dataset Layout

The scripts expect:

```text
dataset/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  metadata.csv
  yolo.yaml
```

`metadata.csv` must contain:

```text
image,split,altitude,category
```

The classifier uses `metadata.csv`. YOLO uses `dataset/yolo.yaml`.

## Classifier

Default run:

```powershell
python train_classifier.py
```

Example with a specific backbone:

```powershell
python train_classifier.py --backbone mobilenetv2_100 --epochs 50 --batch 64 --out runs/classifier_mobilenetv2
```

Useful options:

```text
--backbone       timm backbone name
--epochs         number of training epochs
--batch          batch size
--size           image resize size
--lr             learning rate
--pretrained     use pretrained timm weights
--no-pretrained  train from random initialization
```

Outputs are written to `--out`:

```text
config.json
history.csv
model_best.pt
metrics.json
predictions.csv
```

## YOLO

Default run:

```powershell
python train_yolo.py
```

Example:

```powershell
python train_yolo.py --model yolo11s.pt --epochs 50 --batch 16 --imgsz 640 --out runs/yolo11s
```

Useful options:

```text
--model          YOLO checkpoint, such as yolo11n.pt or yolo11s.pt
--data           YOLO dataset YAML
--epochs         number of training epochs
--batch          batch size
--imgsz          training image size
--device         CUDA device, such as 0
--pretrained     use pretrained model weights
--no-pretrained  disable pretrained initialization
```

YOLO outputs are written under `--out`, including Ultralytics results and `config.json`.
