<div align="center">

# 🏍️ Helmet Detector

### Sistema de detecção de capacetes em motociclistas em tempo real utilizando YOLOv11

Detecção automática de motociclistas utilizando e não utilizando capacete por meio de visão computacional e aprendizado profundo.

**Treinado com PyTorch e Ultralytics YOLOv11, com suporte a GPU, exportação para Core ML e inferência em tempo real.**

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/YOLOv11-Ultralytics-111F68?style=for-the-badge">
  <img src="https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/CoreML-Supported-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

</div>

---

## ✨ Funcionalidades

- 🏍️ Detecção de motociclistas utilizando capacete
- 🚨 Detecção de motociclistas sem capacete
- ⚡ Inferência em tempo real com suporte a GPU
- 🧠 Treinamento utilizando YOLOv11 e PyTorch
- 📦 Exportação para Core ML (.mlpackage)
- 🎯 Classificação baseada em score de confiança
- 📊 Geração de alertas de não conformidade
- 📸 Salvamento automático de evidências
- 🔍 Processamento de imagens, vídeos e diretórios completos
- 🔄 Conversão automática de anotações XML para YOLO

---

## 📌 Visão Geral

| Item | Detalhe |
|--------|--------|
| Tarefa | Object Detection |
| Modelo | YOLO11s |
| Framework | Ultralytics + PyTorch + CUDA |
| Classes | BikesHelmets / BikesNoHelmets |
| Exportação | Core ML |
| Inferência | Tempo Real |
| Hardware | CPU ou GPU NVIDIA |

---

## 🗂️ Estrutura do Projeto

```text
Helmet_detect/
|
├── export_model.py
├── main.py
├── train_helmet_yolo.py
├── xml_to_yolo_converter.py
├── helmet_dataset.yaml
├── README.md
├── LICENSE
|
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   └── labels/
│       ├── xml/
│       └── yolo/
|
└── runs/
    └── detect/
        └── helmet_yolo11s_v2/
            └── weights/
                ├── best.pt
                └── best.mlpackage
````

---

<a id="instalacao"></a>

## ⚙️ Instalação

### Clonar o repositório

```bash
git clone https://github.com/JonathanMar/Helmet_detect.git
cd Helmet_detect
```

### Criar ambiente virtual

```bash
python -m venv env
source env/bin/activate
```

Windows:

```powershell
env\Scripts\activate
```

### Instalar dependências

```bash
pip install torch torchvision torchaudio
pip install ultralytics
```

---

<a id="dataset"></a>

## 🗃️ Dataset

Estrutura esperada:

```text
dataset/
|
├── images/
│   ├── train/
│   └── val/
|
└── labels/
    ├── train/
    └── val/
```

Arquivo de configuração:

```yaml
train: ./dataset/images/train
val: ./dataset/images/val

nc: 2

names:
  - BikesHelmets
  - BikesNoHelmets
```

### Classes

| ID | Classe         |
| -- | -------------- |
| 0  | BikesHelmets   |
| 1  | BikesNoHelmets |

---

## 🔄 Conversão XML → YOLO

Para converter anotações Pascal VOC XML para o formato YOLO:

```bash
python xml_to_yolo_converter.py
```

Classes utilizadas:

```python
classes = [
    "With Helmet",
    "Without Helmet"
]
```

---

<a id="treinamento"></a>

## 🧠 Treinamento

Executar:

```bash
python train_helmet_yolo.py
```

Configuração principal:

```python
model.train(
    data="helmet_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=24,
    name="helmet_yolo11s_v2",
    patience=50,
    device=0,
    workers=8,
    optimizer="AdamW",
    lr0=0.001,
    augment=True,
    cos_lr=True
)
```

### Hiperparâmetros

| Parâmetro     | Valor     |
| ------------- | --------- |
| Modelo        | YOLO11s   |
| Epochs        | 100       |
| Batch Size    | 24        |
| Imagem        | 640x640   |
| Optimizer     | AdamW     |
| Learning Rate | 0.001     |
| Scheduler     | Cosine LR |
| Patience      | 50        |
| Augmentation  | Ativado   |

---

<a id="inferencia"></a>

## 🚀 Inferência

Executar:

```bash
python main.py
```

Fluxo executado:

1. Carrega o modelo treinado
2. Detecta automaticamente GPU ou CPU
3. Processa imagens do diretório configurado
4. Detecta motociclistas sem capacete
5. Gera alertas de não conformidade
6. Salva imagens anotadas

### Exemplo de saída

```text
--- RESULTADOS DA ANALISE DE CONFORMIDADE ---

⚠️ 12 Detecções de 'Sem Capacete' encontradas!

[ALERTA]
Imagem: moto_001.jpg
Confiança: 0.943
BBOX: [120, 85, 312, 410]

[ALERTA]
Imagem: moto_002.jpg
Confiança: 0.918
BBOX: [95, 74, 285, 398]
```

---

## 📊 Classes Monitoradas

```python
CLASSES = {
    0: "BikesHelmets",
    1: "BikesNoHelmets"
}
```

Classe alvo para fiscalização:

```python
TARGET_CLASS_ID = 1
```

---

<a id="exportacao-para-core-ml"></a>

## 📦 Exportação para Core ML

Executar:

```bash
python export_model.py
```

Código utilizado:

```python
from ultralytics import YOLO

model = YOLO(
    "runs/detect/helmet_yolo11s_v2/weights/best.pt"
)

model.export(
    format="coreml",
    int8=True,
    simplify=True
)
```

Resultado:

```text
runs/detect/helmet_yolo11s_v2/weights/best.mlpackage
```

Compatível com:

* iOS
* iPadOS
* macOS
* Apple Silicon

---

## 🎯 Aplicações

* Fiscalização eletrônica
* Segurança viária
* Smart Cities
* Sistemas ITS
* Monitoramento urbano
* Rodovias inteligentes
* Edge AI
* Câmeras de trânsito
* Análise automática de conformidade

---

## 🛠️ Tecnologias Utilizadas

| Categoria           | Tecnologia     |
| ------------------- | -------------- |
| Linguagem           | Python         |
| Deep Learning       | PyTorch        |
| Detecção de Objetos | YOLOv11        |
| Framework           | Ultralytics    |
| Aceleração          | CUDA           |
| Exportação          | Core ML        |
| Anotação            | Pascal VOC XML |
| Dataset             | YOLO Format    |
| Versionamento       | Git + GitHub   |

---

## 📚 Citation

```bibtex
@misc{marcon2026helmetdetector,
  author = {Jonathan Marcon},
  title = {Helmet Detector: Real-Time Motorcycle Helmet Detection Using YOLOv11},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/JonathanMar/Helmet_detect}
}
```

---

## 📄 License

Este projeto está licenciado sob a licença MIT.

Consulte o arquivo LICENSE para mais informações.

---

## 🏁 Conclusão

O Helmet Detector utiliza YOLOv11 para identificar motociclistas com e sem capacete em imagens e vídeos. O sistema oferece treinamento personalizado, inferência em tempo real, exportação para Core ML e suporte a aceleração por GPU, sendo adequado para aplicações de fiscalização automática, monitoramento urbano e sistemas inteligentes de transporte.
