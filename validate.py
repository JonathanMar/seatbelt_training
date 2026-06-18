"""
validate.py — Avaliação do modelo treinado no split de validação.

Uso:
    python validate.py [--model CAMINHO]

Exemplos:
    python validate.py
    python validate.py --model runs/seatbelt_yolov8n/weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

import config as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida o modelo YOLOv8 treinado.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(
            cfg.BASE_DIR / cfg.RUNS_DIR / cfg.NAME / "weights" / "best.pt"
        ),
        help="Caminho para o arquivo .pt do modelo (default: best.pt do run atual).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split do dataset a avaliar (default: val).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERRO] Modelo nao encontrado: {model_path}", file=sys.stderr)
        print("Execute train.py primeiro.", file=sys.stderr)
        sys.exit(1)

    # Garante YAML com path absoluto (corrige bug do Ultralytics)
    data_yaml = cfg.build_dataset_yaml()

    print("=" * 60)
    print(f"  Avaliando : {model_path}")
    print(f"  Split     : {args.split}")
    print(f"  Dataset   : {data_yaml}")
    print("=" * 60)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        device=cfg.DEVICE,
        imgsz=cfg.IMG_SIZE,
        batch=cfg.BATCH,
        workers=cfg.WORKERS,
        verbose=True,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("  RESULTADOS DE VALIDAÇÃO")
    print("-" * 60)
    print(f"  mAP@50        : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95     : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
