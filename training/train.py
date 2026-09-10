"""Training entrypoint — fine-tune AASIST-L on Indian voices (en/hi/kn).

Usage:
    python -m training.train --data data --epochs 12 --out training/weights

Layout expected:
    data/en/bonafide/*.wav   data/en/synthetic/*.wav
    data/hi/bonafide/*.wav   data/hi/synthetic/*.wav
    data/kn/bonafide/*.wav   data/kn/synthetic/*.wav

Produces:
    training/weights/aasist_l_best.pt   — best checkpoint (EER-selected)
    training/weights/aasist_l.onnx      — exported ONNX (float)
    training/weights/aasist_l_int8.onnx — INT8 dynamic-quantized (~4x smaller)

CPU-first: designed for college hardware (no GPU required).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .aasist_model import AASISTL
from .datasets import make_loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voiceshield.train")


def compute_eer(labels: list[int], scores: list[float]) -> float:
    """Equal Error Rate via ROC sweep (label 1 = synthetic positive)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    thresholds = np.unique(scores)
    best = (None, float("inf"))
    for t in thresholds:
        pred = scores >= t
        fpr = float(np.mean(pred[labels == 0]))
        fnr = float(np.mean(~pred[labels == 1]))
        d = abs(fpr - fnr)
        if d < best[1]:
            best = (float((fpr + fnr) / 2), d)
    return best[0] if best[0] is not None else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    labels, scores, loss_sum, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(crit(logits, y)) * len(y)
        n += len(y)
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels += y.cpu().tolist()
        scores += probs.cpu().tolist()
    return {"loss": loss_sum / max(n, 1), "eer": compute_eer(labels, scores)}


def export_onnx(model: AASISTL, out_path: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 1, 80, 380)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["logmel"],
        output_names=["logits"],
        dynamic_axes={"logmel": {3: "time"}, "logits": {0: "batch"}},
        opset_version=15,
    )
    log.info("ONNX exported -> %s", out_path)


def quantize(src: Path, dst: Path) -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        log.info("INT8 quantized -> %s (%.1fx smaller)", dst, src.stat().st_size / max(dst.stat().st_size, 1))
    except Exception as exc:
        log.warning("quantization skipped: %s", exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="training/weights")
    ap.add_argument("--langs", default="en,hi,kn")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)

    train_loader, val_loader = make_loaders(args.data, batch_size=args.batch_size)
    log.info("train=%d val=%d", len(train_loader.dataset), len(val_loader.dataset))

    model = AASISTL().to(device)
    log.info("AASIST-L params: %s", model.num_params())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_eer = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += float(loss) * len(y)
        sched.step()
        train_loss = run_loss / len(train_loader.dataset)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), **{k: round(v, 4) for k, v in val.items()}})
        log.info("epoch %d | train_loss %.4f | val_loss %.4f | EER %.4f", epoch, train_loss, val["loss"], val["eer"])

        if val["eer"] < best_eer:
            best_eer = val["eer"]
            torch.save(model.state_dict(), out_dir / "aasist_l_best.pt")
            log.info("  ↳ new best (EER %.4f) saved", best_eer)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    # reload best and export
    model.load_state_dict(torch.load(out_dir / "aasist_l_best.pt", map_location=device))
    export_onnx(model, out_dir / "aasist_l.onnx")
    quantize(out_dir / "aasist_l.onnx", out_dir / "aasist_l_int8.onnx")
    log.info("done. best EER=%.4f — copy aasist_l_int8.onnx to backend/inference/model/", best_eer)


if __name__ == "__main__":
    main()
