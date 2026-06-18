"""
predict.py — Inferencia do detector de cinto de seguranca com YOLOv8.

Suporta:
  - Imagem unica  (.jpg, .jpeg, .png, .bmp, .webp)
  - Diretorio com imagens
  - Video         (.mp4, .avi, .mov, .mkv)
  - Webcam        (--source 0)
  - URL           (http/https)

Uso:
    # Imagem
    python predict.py --source foto.jpg

    # Video
    python predict.py --source video.mp4

    # Webcam ao vivo
    python predict.py --source 0 --show

    # Pasta de imagens
    python predict.py --source dataset/test/images/

    # Modelo e confianca customizados
    python predict.py --source foto.jpg --model runs/detect/seatbelt_yolov8n/weights/best.pt --conf 0.4

Resultados salvos em runs/predict/<nome>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

import config as cfg

# Extensoes suportadas
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detector de cinto de seguranca — inferencia com YOLOv8.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Origem da inferencia: caminho de imagem, video, diretorio, "
            "indice de webcam (0, 1...) ou URL."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(cfg.BASE_DIR / cfg.RUNS_DIR / cfg.NAME / "weights" / "best.pt"),
        help="Caminho para o modelo .pt treinado (default: best.pt do run atual).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confianca minima para considerar uma deteccao (default: 0.45).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="Threshold de IoU para NMS (default: 0.50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=cfg.IMG_SIZE,
        help=f"Tamanho de inferencia (default: {cfg.IMG_SIZE}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Exibir resultado em janela ao vivo (requer display / webcam).",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        default=False,
        help="Salvar anotacoes em formato YOLO .txt.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        default=False,
        help="Incluir scores de confianca nos .txt salvos.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Nome do subdiretorio de saida em runs/predict/ (default: exp).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(cfg.DEVICE),
        help=f"Device: 0, cpu, cuda:0 etc. (default: {cfg.DEVICE}).",
    )

    return parser.parse_args()


def _detect_source_type(source: str) -> str:
    """Classifica o tipo de fonte para exibir no log."""
    src = source.strip()

    if src.isdigit():
        return f"webcam (device {src})"

    if src.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return "url/stream"

    p = Path(src)
    if p.is_dir():
        return f"diretorio ({len(list(p.iterdir()))} entradas)"

    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        return "imagem"
    if ext in VIDEO_EXTS:
        return "video"

    return "desconhecido"


def _validate(args: argparse.Namespace) -> None:
    """Valida model e source antes de iniciar."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"[ERRO] Modelo nao encontrado: {model_path}\n"
            "       Execute 'python train.py' primeiro.",
            file=sys.stderr,
        )
        sys.exit(1)

    src = args.source.strip()
    if not src.isdigit() and not src.startswith(("http", "rtsp", "rtmp")):
        p = Path(src)
        if not p.exists():
            print(f"[ERRO] Source nao encontrado: {p}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    args = parse_args()
    _validate(args)

    src_type = _detect_source_type(args.source)

    print("=" * 60)
    print("  SEATBELT DETECTOR — Inferencia")
    print("=" * 60)
    print(f"  Modelo  : {args.model}")
    print(f"  Source  : {args.source}  [{src_type}]")
    print(f"  Conf    : {args.conf}   |  IoU: {args.iou}")
    print(f"  Device  : {args.device}")
    print(f"  Saida   : runs/predict/{args.name}/")
    print("=" * 60)

    model = YOLO(args.model)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        show=args.show,
        save=True,                      # salva sempre a midia anotada
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project="runs/predict",
        name=args.name,
        exist_ok=True,
        stream=True,                    # stream=True: eficiente para video/webcam
        verbose=False,
    )

    # Itera sobre os resultados (necessario quando stream=True)
    n_frames   = 0
    n_with     = 0
    n_without  = 0

    for result in results:
        n_frames += 1
        boxes = result.boxes
        if boxes is not None:
            for cls_id in boxes.cls.tolist():
                cls_id = int(cls_id)
                if cls_id == 1:      # person_with_seatbelt
                    n_with += 1
                elif cls_id == 0:    # person_without_seatbelt
                    n_without += 1

    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO DAS DETECCOES")
    print("-" * 60)
    print(f"  Frames / imagens processados : {n_frames}")
    print(f"  Pessoas COM cinto            : {n_with}")
    print(f"  Pessoas SEM cinto            : {n_without}")
    print(f"  Total de deteccoes           : {n_with + n_without}")
    print(f"  Resultados salvos em         : runs/predict/{args.name}/")
    print("=" * 60)

    if n_without > 0:
        pct = n_without / (n_with + n_without) * 100
        print(f"\n  ALERTA: {n_without} deteccoes sem cinto ({pct:.1f}% do total)!")


if __name__ == "__main__":
    main()
