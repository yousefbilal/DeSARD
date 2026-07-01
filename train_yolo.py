import argparse
import json
import os
from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "yolo.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "yolo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str((out.parent / ".ultralytics").resolve()))


    model = YOLO(args.model)
    kwargs = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "project": str(out.parent),
        "name": out.name,
        "exist_ok": True,
        "pretrained": args.pretrained,
        "plots": True,
        "verbose": True,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    (out / "config.json").write_text(json.dumps({
        "data": str(args.data.resolve()),
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "device": args.device,
        "pretrained": args.pretrained,
    }, indent=2), encoding="utf-8")

    model.train(**kwargs)


if __name__ == "__main__":
    main()
