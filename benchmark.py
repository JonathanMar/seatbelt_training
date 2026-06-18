"""
benchmark.py — Encontra o maior batch seguro para a GPU disponível.

Executa 1 época de treinamento para cada tamanho de batch candidato e para
assim que o primeiro OOM (Out Of Memory) é detectado.

Uso:
    python benchmark.py
"""

from __future__ import annotations

import torch
from ultralytics import YOLO

import config as cfg

# Batches candidatos a testar (em ordem crescente)
BATCH_CANDIDATES: list[int] = [8, 16, 24, 32, 48, 64]


def main() -> None:
    # Garante YAML com path absoluto antes de qualquer treino
    data_yaml = cfg.build_dataset_yaml()

    print("=" * 60)
    print("  BENCHMARK DE BATCH SIZE")
    print(f"  GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
          if torch.cuda.is_available() else "  VRAM : N/A")
    print("=" * 60)

    last_ok: int | None = None

    for batch in BATCH_CANDIDATES:
        print(f"\n[TEST] batch={batch}")

        # Reinicia o modelo a cada iteracao para evitar estado residual
        model = YOLO(cfg.MODEL)

        try:
            model.train(
                data=str(data_yaml),
                epochs=1,
                imgsz=cfg.IMG_SIZE,
                batch=batch,          # ← variável correta (bug corrigido)
                device=cfg.DEVICE,
                amp=cfg.AMP,
                cache=False,          # sem cache para medir uso de VRAM limpo
                workers=cfg.WORKERS,
                verbose=False,
                plots=False,
                save=False,
                exist_ok=True,
                project="runs",
                name="_benchmark_tmp",
            )

            print(f"  [OK] batch={batch} funcionou")
            last_ok = batch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                reserved  = torch.cuda.memory_reserved(0) / 1024 ** 3
                print(f"       VRAM alocada : {allocated:.2f} GB")
                print(f"       VRAM reservada: {reserved:.2f} GB")

            torch.cuda.empty_cache()

        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            print(f"  [FAIL] batch={batch} → OOM ou erro: {exc}")
            torch.cuda.empty_cache()
            break

    print("\n" + "=" * 60)
    if last_ok is not None:
        print(f"  Maior batch seguro encontrado: {last_ok}")
        print(f"  Atualize BATCH = {last_ok} em config.py")
    else:
        print("  Nenhum batch funcionou. Verifique a GPU / drivers.")
    print("=" * 60)


if __name__ == "__main__":
    main()
