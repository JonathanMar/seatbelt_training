# Seatbelt Detector - YOLOv8 Training Pipeline

Pipeline completo de treinamento, validacao e exportacao de um detector de
cinto de seguranca usando **YOLOv8n** (Ultralytics).

---

## Sumario

1. [Visao geral](#visao-geral)
2. [Estrutura do projeto](#estrutura-do-projeto)
3. [Hardware testado](#hardware-testado)
4. [Requisitos](#requisitos)
5. [Instalacao](#instalacao)
6. [Dataset](#dataset)
7. [Configuracao](#configuracao)
8. [Uso](#uso)
9. [Hiperparametros](#hiperparametros)
10. [Resultados esperados](#resultados-esperados)
11. [Exportacao para producao](#exportacao-para-producao)
12. [Boas praticas aplicadas](#boas-praticas-aplicadas)

---

## Visao geral

| Item | Detalhe |
|---|---|
| Tarefa | Deteccao de objetos (Object Detection) |
| Modelo | YOLOv8n (nano, 3.2 M parametros) |
| Framework | Ultralytics 8.4.35 + PyTorch 2.11 (CUDA 13.0) |
| Classes | `person_without_seatbelt` / `person_with_seatbelt` |
| Formato de exportacao | ONNX (opset 17, batch dinamico) |

---

## Estrutura do projeto

```
seatbelt_training/
|
|-- config.py                  # Fonte unica de verdade: hiperparametros e caminhos
|-- train.py                   # Script principal de treinamento
|-- validate.py                # Avaliacao de metricas (mAP, Precision, Recall)
|-- export_onnx.py             # Exportacao do modelo para ONNX
|-- benchmark.py               # Busca automatica do maior batch seguro
|-- requirements.txt           # Dependencias Python com versoes minimas
|
|-- dataset/
|   |-- data.yaml              # Template original do dataset
|   |-- _data_resolved.yaml    # Gerado em runtime com paths absolutos (auto)
|   |-- train/
|   |   |-- images/            # 5.840 imagens de treino
|   |   `-- labels/            # Anotacoes YOLO (.txt)
|   |-- valid/
|   |   |-- images/            # 1.110 imagens de validacao
|   |   `-- labels/
|   `-- test/
|       |-- images/            # 306 imagens de teste
|       `-- labels/
|
`-- runs/
    `-- seatbelt_yolov8n/      # Criado automaticamente pelo treinamento
        |-- weights/
        |   |-- best.pt        # Melhor checkpoint (mAP50-95)
        |   `-- last.pt        # Ultimo checkpoint
        `-- ...                # Graficos, metricas, confusion matrix
```

> **Nota sobre `_data_resolved.yaml`:**
> O Ultralytics resolve o campo `path:` do data.yaml relativo ao CWD, nao ao
> diretorio do YAML. Para evitar erros de "images not found", o `config.py`
> gera este arquivo com o caminho absoluto correto em cada execucao.

---

## Hardware testado

| Componente | Especificacao |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER |
| VRAM | 7.6 GB (Turing, sm_75) |
| CPU | Intel Xeon E5-2666 v3 (10 cores / 20 threads) |
| Armazenamento | SSD (cache em disco habilitado) |
| OS | Ubuntu / Pop!_OS (Linux) |

> Para outras GPUs, ajuste `BATCH` em `config.py` ou execute `python benchmark.py`
> para encontrar o batch maximo automaticamente.

---

## Requisitos

- Python **3.10+**
- CUDA **12.x** ou superior (para GPU)
- pip / ambiente virtual

---

## Instalacao

```bash
# 1. Clonar o repositorio
git clone <url-do-repositorio>
cd seatbelt_training

# 2. Criar e ativar ambiente virtual
python3 -m venv env
source env/bin/activate

# 3. Instalar PyTorch com CUDA
#    Verifique a versao correta em: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Instalar demais dependencias
pip install -r requirements.txt
```

> **Windows:** substitua `source env/bin/activate` por `env\Scripts\activate`

---

## Dataset

### Formato YOLO

O dataset segue o formato padrao YOLO: um arquivo `.txt` de anotacao por imagem,
com cada linha representando uma caixa delimitadora:

```
<class_id> <cx> <cy> <width> <height>
```

Todos os valores em coordenadas normalizadas (0.0 a 1.0).

### Classes

| ID | Nome | Descricao |
|---|---|---|
| 0 | `person_without_seatbelt` | Pessoa detectada SEM cinto de seguranca |
| 1 | `person_with_seatbelt` | Pessoa detectada COM cinto de seguranca |

### Estatisticas do dataset

| Split | Imagens |
|---|---|
| Treino | 5.840 |
| Validacao | 1.110 |
| Teste | 306 |
| **Total** | **7.256** |

### Adicionando seus proprios dados

1. Coloque imagens em `dataset/train/images/` e anotacoes em `dataset/train/labels/`
2. Repita para `valid/` e `test/`
3. Ajuste `nc` e `names` em `config.py` (funcao `build_dataset_yaml`) se as classes mudarem
4. Nao e necessario editar `data.yaml` manualmente

---

## Configuracao

Toda a configuracao esta centralizada em `config.py`.
**Nenhum hiperparametro esta hardcoded nos scripts de treino.**

### Parametros principais

```python
# Modelo base (pode ser trocado por yolov8s, yolov8m etc.)
MODEL = "yolov8n.pt"

# Batch — RTX 2060 SUPER (7.6 GB): batch 16 e seguro com YOLOv8n@640
# Use -1 para AutoBatch (YOLO encontra o maximo automaticamente)
BATCH = 16

# Epocas e Early Stopping
EPOCHS   = 150
PATIENCE = 30    # para se nao houver melhoria em 30 epocas consecutivas

# Cache — "disk": recomendado para SSD | True: RAM | False: sem cache
CACHE = "disk"

# Mixed Precision (FP16) — Tensor Cores do Turing
AMP = True
```

---

## Uso

### Treinar

```bash
python train.py
```

O script:
1. Detecta GPU e exibe informacoes de hardware
2. Gera `_data_resolved.yaml` com paths absolutos
3. Inicia o treinamento com os hiperparametros de `config.py`
4. Salva o melhor modelo em `runs/seatbelt_yolov8n/weights/best.pt`

---

### Validar

```bash
# Avaliar no split de validacao (padrao)
python validate.py

# Avaliar no split de teste
python validate.py --split test

# Avaliar um modelo especifico
python validate.py --model runs/seatbelt_yolov8n/weights/last.pt

# Combinar opcoes
python validate.py --model runs/seatbelt_yolov8n/weights/best.pt --split test
```

Saida esperada:

```
============================================================
  RESULTADOS DE VALIDACAO
------------------------------------------------------------
  mAP@50        : 0.9241
  mAP@50-95     : 0.7183
  Precision     : 0.9105
  Recall        : 0.8997
============================================================
```

---

### Exportar para ONNX

```bash
# Exportar best.pt com configuracoes padrao
python export_onnx.py

# Personalizar modelo, resolucao e opset
python export_onnx.py --model runs/seatbelt_yolov8n/weights/best.pt --imgsz 640 --opset 17
```

O arquivo `.onnx` e salvo no mesmo diretorio do `.pt`.

---

### Benchmark de batch

```bash
python benchmark.py
```

Testa automaticamente os batches `[8, 16, 24, 32, 48, 64]` por 1 epoca cada
e para no primeiro OOM, reportando o maior valor seguro:

```
============================================================
  Maior batch seguro encontrado: 32
  Atualize BATCH = 32 em config.py
============================================================
```

---

## Hiperparametros

### Otimizador (AdamW + Cosine Annealing)

| Parametro | Valor | Descricao |
|---|---|---|
| `optimizer` | AdamW | Convergencia rapida com regularizacao implicita |
| `lr0` | 1e-3 | Learning rate inicial |
| `lrf` | 0.01 | Fator de decay: LR_final = lr0 x lrf = **1e-5** |
| `weight_decay` | 5e-4 | Regularizacao L2 |
| `momentum` | 0.937 | Momento do otimizador |
| `cos_lr` | True | Cosine annealing scheduler |
| `warmup_epochs` | 3.0 | Epocas de aquecimento com LR crescente |
| `warmup_momentum` | 0.8 | Momento durante o warmup |
| `warmup_bias_lr` | 0.1 | LR dos biases durante o warmup |

### Augmentacao

| Parametro | Valor | Descricao |
|---|---|---|
| `mosaic` | 1.0 | Combina 4 imagens — diversidade de contexto |
| `mixup` | 0.10 | Mistura pares de imagens — regularizacao |
| `copy_paste` | 0.10 | Cola objetos entre imagens — melhora recall |
| `close_mosaic` | 10 | Desliga mosaic nas ultimas 10 epocas |
| `hsv_h/s/v` | 0.015 / 0.7 / 0.4 | Variacao de matiz, saturacao e brilho |
| `degrees` | 5.0 | Rotacao aleatoria em graus |
| `translate` | 0.10 | Translacao aleatoria (10% da imagem) |
| `scale` | 0.50 | Zoom aleatorio ate 50% |
| `shear` | 2.0 | Distorcao de perspectiva (graus) |
| `fliplr` | 0.5 | Flip horizontal com prob. 50% |
| `flipud` | 0.0 | Flip vertical desativado |
| `label_smoothing` | 0.0 | Desativado; use 0.1 para datasets com ruido |

---

## Resultados esperados

Com os 5.840 exemplos de treino e os hiperparametros configurados:

| Metrica | Referencia tipica |
|---|---|
| mAP@50 | > 0.90 |
| mAP@50-95 | > 0.65 |
| Precision | > 0.88 |
| Recall | > 0.85 |

> Os valores reais dependem da qualidade, diversidade e balanceamento do dataset.

---

## Exportacao para producao

Apos exportar com `python export_onnx.py`, o modelo pode ser usado com:

### ONNX Runtime (Python)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "best.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: frame.astype(np.float32)})
```

### OpenCV DNN

```python
import cv2
net = cv2.dnn.readNetFromONNX("best.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

### Ultralytics (diretamente)

```python
from ultralytics import YOLO
model = YOLO("best.onnx")
results = model.predict("frame.jpg", conf=0.5)
```

---

## Boas praticas aplicadas

| Pratica | Implementacao |
|---|---|
| Fonte unica de verdade | Todos os hiperparametros em `config.py` |
| Paths absolutos em runtime | `build_dataset_yaml()` corrige bug do Ultralytics |
| Fail-fast | Paths e arquivos validados antes de iniciar qualquer operacao |
| CLI com argparse | `validate.py` e `export_onnx.py` aceitam argumentos |
| Reproducibilidade | `seed=42` fixado em todas as execucoes |
| Performance GPU | `cudnn.benchmark=True`, AMP (FP16), cache em disco |
| Early stopping | Para automaticamente se nao houver melhoria em 30 epocas |
| Type hints | Todos os modulos com anotacoes (`from __future__ import annotations`) |
| Docstrings | Todos os modulos e funcoes documentados |
| Encapsulamento | Logica em funcoes; `if __name__ == "__main__"` em todos os scripts |

---

## Inferencia (Teste do Modelo)

Apos o treinamento, use `predict.py` para testar o modelo em imagens,
videos, webcam ou pastas inteiras. Os resultados sao salvos em `runs/predict/`.

### Instalacao rapida

```bash
source env/bin/activate
```

### Imagem

```bash
# Imagem unica
python predict.py --source foto.jpg

# Confianca customizada
python predict.py --source foto.jpg --conf 0.4

# Salvar anotacoes YOLO .txt junto com a imagem
python predict.py --source foto.jpg --save-txt
```

Resultado salvo em `runs/predict/exp/foto.jpg` com as bounding boxes desenhadas.

### Video

```bash
# Video MP4
python predict.py --source video.mp4

# Com nome de saida customizado
python predict.py --source video.mp4 --name teste_video
```

Resultado salvo em `runs/predict/teste_video/video.mp4` com deteccoes frame a frame.

### Webcam ao vivo

```bash
# Webcam padrao (device 0) com janela ao vivo
python predict.py --source 0 --show

# Segunda webcam
python predict.py --source 1 --show
```

### Pasta de imagens

```bash
# Todas as imagens do split de teste
python predict.py --source dataset/test/images/ --name batch_test
```

### Argumentos disponiveis

| Argumento | Padrao | Descricao |
|---|---|---|
| `--source` | obrigatorio | Imagem, video, diretorio, 0 (webcam) ou URL |
| `--model` | `best.pt` do run atual | Caminho para o modelo `.pt` |
| `--conf` | `0.45` | Confianca minima para uma deteccao ser considerada |
| `--iou` | `0.50` | Threshold IoU para NMS |
| `--imgsz` | `640` | Tamanho de inferencia |
| `--show` | `False` | Exibe janela ao vivo (requer display) |
| `--save-txt` | `False` | Salva anotacoes em formato YOLO `.txt` |
| `--save-conf` | `False` | Inclui score de confianca nos `.txt` |
| `--name` | `exp` | Nome do subdiretorio em `runs/predict/` |
| `--device` | GPU (se disponivel) | Device: `0`, `cpu`, `cuda:0` |

### Saida do terminal

```
============================================================
  SEATBELT DETECTOR - Inferencia
============================================================
  Modelo  : runs/detect/seatbelt_yolov8n/weights/best.pt
  Source  : video.mp4  [video]
  Conf    : 0.45   |  IoU: 0.50
  Device  : 0
  Saida   : runs/predict/exp/
============================================================

============================================================
  RESUMO DAS DETECCOES
------------------------------------------------------------
  Frames / imagens processados : 450
  Pessoas COM cinto            : 312
  Pessoas SEM cinto            : 47
  Total de deteccoes           : 359
  Resultados salvos em         : runs/predict/exp/
============================================================

  ALERTA: 47 deteccoes sem cinto (13.1% do total)!
```
