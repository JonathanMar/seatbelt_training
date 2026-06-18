"""
config.py — Configuracao centralizada do pipeline de treinamento.

Hardware alvo:
  GPU : NVIDIA RTX 2060 SUPER (7.6 GB VRAM, Turing sm_75)
  CPU : Intel Xeon E5-2666 v3 (10c / 20t)
  Cache: SSD

Modelo: YOLOv8n  (nano — velocidade + eficiencia para VRAM limitada)
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = BASE_DIR / "dataset"

# YOLO resolve o campo 'path' do data.yaml relativo ao CWD (nao ao YAML),
# o que causa 'images not found' quando o script e executado de outro diretorio.
# Solucao: gerar um YAML temporario com o caminho absoluto em runtime.
_DATASET_YAML_TEMPLATE: Path = DATASET_DIR / "data.yaml"
DATASET: Path = DATASET_DIR / "_data_resolved.yaml"


def build_dataset_yaml() -> Path:
    """
    Gera (ou atualiza) o arquivo YAML do dataset com caminhos absolutos,
    evitando o bug de resolucao relativa ao CWD do Ultralytics.

    Retorna o Path do YAML gerado.
    """
    resolved = {
        "path": str(DATASET_DIR),          # absoluto → YOLO nunca erra
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc": 2,
        "names": {
            0: "person_without_seatbelt",
            1: "person_with_seatbelt",
        },
    }

    with open(DATASET, "w", encoding="utf-8") as f:
        yaml.dump(resolved, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return DATASET


# ---------------------------------------------------------------------------
# Modelo base
# ---------------------------------------------------------------------------
MODEL: str = "yolov8n.pt"  # nano — excelente para fine-tuning com VRAM ~8 GB

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
DEVICE: int | str = 0 if torch.cuda.is_available() else "cpu"

# AMP: Mixed Precision (FP16) → Tensor Cores do Turing; desligar so se houver
# instabilidade numerica (NaN loss).
AMP: bool = True

# Workers de DataLoader.
#
# IMPORTANTE — PyTorch 2.11+cu130 + Linux: a thread interna de pin_memory
# falha com 'cudaErrorOperatingSystem' quando workers > 0, pois o driver
# CUDA 13.0 nao suporta cudaHostAlloc() nesse ambiente.
# SOLUCAO: workers=0 executa o DataLoader no processo principal, eliminando
# a thread de pin_memory e o crash. Custo: leitura de dados ligeiramente
# mais lenta, mas o gargalo real e a GPU (nao a CPU/IO).
# Se atualizar o driver ou o kernel, teste com workers=4 ou workers=8.
WORKERS: int = 0

# Cache em disco evita releitura de imagem a cada epoca (recomendado quando o
# dataset cabe no SSD e nao na RAM).
CACHE: str = "disk"  # alternativas: True (RAM) | False (sem cache)

# ---------------------------------------------------------------------------
# Imagens e batch
# ---------------------------------------------------------------------------
IMG_SIZE: int = 640

# RTX 2060 SUPER: 7.6 GB — batch 16 e seguro com YOLOv8n@640.
# Use -1 para AutoBatch (YOLO detecta automaticamente o maximo seguro).
BATCH: int = 16

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
EPOCHS: int = 150

# Paciencia para Early Stopping. Com 150 epocas, 30 e razoavel.
PATIENCE: int = 30

# Warmup: nr de epocas com LR crescente (evita instabilidade no inicio).
WARMUP_EPOCHS: float = 3.0
WARMUP_MOMENTUM: float = 0.8
WARMUP_BIAS_LR: float = 0.1

# ---------------------------------------------------------------------------
# Otimizador (AdamW)
# ---------------------------------------------------------------------------
OPTIMIZER: str = "AdamW"
LR0: float = 1e-3        # LR inicial
LRF: float = 0.01        # fator de decay final: LR_final = LR0 * LRF
                          # → 1e-3 x 0.01 = 1e-5  (decai adequadamente)
WEIGHT_DECAY: float = 5e-4
MOMENTUM: float = 0.937
COS_LR: bool = True       # scheduler cosine annealing

# ---------------------------------------------------------------------------
# Augmentacao
# ---------------------------------------------------------------------------
HSV_H: float = 0.015
HSV_S: float = 0.7
HSV_V: float = 0.4

DEGREES: float = 5.0
TRANSLATE: float = 0.10
SCALE: float = 0.50
SHEAR: float = 2.0
PERSPECTIVE: float = 0.0

FLIPLR: float = 0.5
FLIPUD: float = 0.0

MOSAIC: float = 1.0
MIXUP: float = 0.10
COPY_PASTE: float = 0.10   # cola objetos de outras imagens → +recall em cenarios densos
CLOSE_MOSAIC: int = 10     # desliga mosaic nas ultimas N epocas para estabilizar

# ---------------------------------------------------------------------------
# Saida
# ---------------------------------------------------------------------------
# Ultralytics 8.4.x insere automaticamente o tipo de task (detect/) entre o
# project e o name. O settings.json ja define runs_dir='runs', entao setar
# project='runs' resultaria em 'runs/detect/runs/name' (duplo).
# Solucao: nao passar project= no model.train(); o YOLO usa o default
# correto de 'runs/detect/{name}'. RUNS_DIR reflete esse caminho real.
NAME: str = "seatbelt_yolov8n"
RUNS_DIR: str = "runs/detect"  # path real gerado pelo Ultralytics — nao usar como argumento
SEED: int = 42
