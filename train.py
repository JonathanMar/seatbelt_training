"""
train.py — Treinamento do detector de cinto de segurança com YOLOv8.

Uso:
    python train.py

O script detecta automaticamente GPU/CPU, exibe informações de hardware e
dispara o treinamento com os hiperparâmetros definidos em config.py.
"""

from __future__ import annotations

import multiprocessing
import sys

import torch
from ultralytics import YOLO

import config as cfg


def _print_hardware_info() -> None:
    """Exibe informações de hardware disponível."""
    print("=" * 60)
    print("  SEATBELT DETECTOR — YOLOv8 Training")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {vram_gb:.2f} GB")
        print(f"  SM     : {props.multi_processor_count} (sm_{props.major}{props.minor})")
        # Habilita acumulação TF32 para operações de matmul (Ampere+) e cuDNN.
        # Em Turing (sm_75) isso é no-op para matmul, mas ativo para cuDNN.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True   # otimiza kernels para shapes fixos
    else:
        print("  GPU    : não disponível — treinando em CPU (lento!)")

    print(f"  Device : {cfg.DEVICE}")
    print(f"  Workers: {cfg.WORKERS}")
    print(f"  Dataset: {cfg.DATASET}")
    print("=" * 60)


def _validate_paths() -> None:
    """Valida paths essenciais e gera o YAML do dataset com caminhos absolutos."""
    if not cfg.DATASET_DIR.exists():
        print(f"[ERRO] Pasta do dataset nao encontrada: {cfg.DATASET_DIR}", file=sys.stderr)
        sys.exit(1)

    # Gera _data_resolved.yaml com path absoluto — corrige bug do Ultralytics
    # que resolve 'path:' relativo ao CWD em vez do diretorio do YAML.
    cfg.build_dataset_yaml()
    print(f"  YAML     : {cfg.DATASET}")


def main() -> None:
    _print_hardware_info()
    _validate_paths()

    model = YOLO(cfg.MODEL)

    model.train(
        # ── Dataset ─────────────────────────────────────────────────────────
        data=str(cfg.DATASET),

        # ── Hardware ────────────────────────────────────────────────────────
        device=cfg.DEVICE,
        amp=cfg.AMP,
        workers=cfg.WORKERS,
        cache=cfg.CACHE,

        # ── Imagens ─────────────────────────────────────────────────────────
        imgsz=cfg.IMG_SIZE,
        batch=cfg.BATCH,

        # ── Treinamento ─────────────────────────────────────────────────────
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,

        # ── Warmup ──────────────────────────────────────────────────────────
        warmup_epochs=cfg.WARMUP_EPOCHS,
        warmup_momentum=cfg.WARMUP_MOMENTUM,
        warmup_bias_lr=cfg.WARMUP_BIAS_LR,

        # ── Otimizador ──────────────────────────────────────────────────────
        optimizer=cfg.OPTIMIZER,
        lr0=cfg.LR0,
        lrf=cfg.LRF,
        weight_decay=cfg.WEIGHT_DECAY,
        momentum=cfg.MOMENTUM,
        cos_lr=cfg.COS_LR,

        # ── Augmentação ─────────────────────────────────────────────────────
        hsv_h=cfg.HSV_H,
        hsv_s=cfg.HSV_S,
        hsv_v=cfg.HSV_V,
        degrees=cfg.DEGREES,
        translate=cfg.TRANSLATE,
        scale=cfg.SCALE,
        shear=cfg.SHEAR,
        perspective=cfg.PERSPECTIVE,
        fliplr=cfg.FLIPLR,
        flipud=cfg.FLIPUD,
        mosaic=cfg.MOSAIC,
        mixup=cfg.MIXUP,
        copy_paste=cfg.COPY_PASTE,
        close_mosaic=cfg.CLOSE_MOSAIC,

        # ── Misc ────────────────────────────────────────────────────────────
        seed=cfg.SEED,
        deterministic=False,  # True → reprodutível mas ~30% mais lento

        # ── Saida ────────────────────────────────────────────────────────────
        # Nao passar project= aqui: Ultralytics 8.4.x ja define o default
        # correto como runs/detect/{name}. Setar project='runs' causaria
        # o path duplicado runs/detect/runs/{name}.
        name=cfg.NAME,
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
    )

    best = cfg.BASE_DIR / cfg.RUNS_DIR / cfg.NAME / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"  Treinamento concluído!")
    print(f"  Melhor modelo salvo em: {best}")
    print("=" * 60)


if __name__ == "__main__":
    # freeze_support é necessário apenas no Windows com executáveis empacotados.
    # Em Linux não tem efeito, mas mantemos para portabilidade.
    multiprocessing.freeze_support()
    main()
