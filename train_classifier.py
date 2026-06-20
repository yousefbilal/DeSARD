import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parent


class DeSARDDataset(Dataset):
    def __init__(self, root: Path, split: str, transform):
        self.root = root
        self.split = split
        self.transform = transform
        metadata = pd.read_csv(root / "metadata.csv")
        metadata = metadata[metadata["split"] == split].copy()
        metadata["target"] = metadata["category"].map({"without human": 0, "human": 1})
        self.rows = metadata[["image", "target"]].to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / "images" / self.split / row["image"]).convert("RGB")
        return self.transform(image), int(row["target"]), row["image"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(size: int):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = fp = fn = tn = 0
    rows = []
    with torch.no_grad():
        for images, targets, names in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * int(targets.numel())
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
            tp += int(((preds == 1) & (targets == 1)).sum().item())
            fp += int(((preds == 1) & (targets == 0)).sum().item())
            fn += int(((preds == 0) & (targets == 1)).sum().item())
            tn += int(((preds == 0) & (targets == 0)).sum().item())
            for name, target, pred, prob in zip(names, targets.cpu(), preds.cpu(), probs.cpu()):
                rows.append({
                    "image": name,
                    "true": int(target.item()),
                    "pred": int(pred.item()),
                    "prob_human": float(prob.item()),
                })
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_human = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if precision_neg + recall_neg else 0.0
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "macro_f1": (f1_human + f1_neg) / 2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "classifier")
    parser.add_argument("--backbone", default="mobilenetv2_100")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, eval_tf = make_transforms(args.size)
    train_ds = DeSARDDataset(args.data, "train", train_tf)
    val_ds = DeSARDDataset(args.data, "val", eval_tf)
    test_ds = DeSARDDataset(args.data, "test", eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = timm.create_model(args.backbone, pretrained=args.pretrained, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []

    (out / "config.json").write_text(json.dumps({
        "data": str(args.data.resolve()),
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "size": args.size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(targets.numel())
            train_total += int(targets.numel())
            train_correct += int((logits.argmax(dim=1) == targets).sum().item())
        val = evaluate(model, val_loader, loss_fn, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(record)
        print(json.dumps(record))
        if val["accuracy"] > best_accuracy:
            best_accuracy = val["accuracy"]
            torch.save({"model": model.state_dict(), "config": json.loads((out / "config.json").read_text())}, out / "model_best.pt")

    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(out / "model_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, loss_fn, device)
    metrics = {key: value for key, value in test.items() if key != "rows"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true", "pred", "prob_human"])
        writer.writeheader()
        writer.writerows(test["rows"])


if __name__ == "__main__":
    main()
