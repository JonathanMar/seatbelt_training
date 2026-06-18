"""
export_onnx.py — Exporta o modelo treinado para ONNX (deploy / inferência).

Uso:
    python export_onnx.py [--model CAMINHO] [--imgsz N] [--opset N]

Exemplos:
    python export_onnx.py
    python export_onnx.py --model runs/seatbelt_yolov8n/weights/best.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

import config as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta modelo YOLOv8 para ONNX.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(
            cfg.BASE_DIR / cfg.RUNS_DIR / cfg.NAME / "weights" / "best.pt"
        ),
        help="Caminho para o arquivo .pt (default: best.pt do run atual).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=cfg.IMG_SIZE,
        help=f"Tamanho da imagem de entrada (default: {cfg.IMG_SIZE}).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Versão do opset ONNX (default: 17).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Exporta com batch size dinâmico (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERRO] Modelo não encontrado: {model_path}", file=sys.stderr)
        print("Execute train.py primeiro.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"  Exportando : {model_path}")
    print(f"  Formato    : ONNX (opset {args.opset})")
    print(f"  imgsz      : {args.imgsz}")
    print(f"  dynamic    : {args.dynamic}")
    print("=" * 60)

    model = YOLO(str(model_path))

    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,    # ONNX Simplifier → remove nós redundantes
        dynamic=args.dynamic,
    )

    print("\n" + "=" * 60)
    print(f"  Exportação concluída!")
    print(f"  Arquivo ONNX: {exported_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
