# AACI - Ambient-Agentic Clinical Intelligence

Ambient-Agentic Clinical Intelligence for Voither HealthOS

This repository contains worker containers running Whisper Large 3, specialized medical and psychiatric libraries (especially in Portuguese), in the process of fine-tuning with a proprietary dataset of over 400 hours of selected audio.

## 🏥 Overview

AACI is a clinical intelligence system designed specifically for the Brazilian healthcare market, providing:

- **Whisper Large 3 Integration**: State-of-the-art speech recognition optimized for medical contexts
- **Medical Vocabulary**: Extensive Portuguese medical and psychiatric terminology
- **Fine-tuning Pipeline**: Custom training infrastructure for domain adaptation
- **Worker Architecture**: Scalable containerized transcription services
- **Clinical Focus**: Specialized for medical consultations and psychiatric evaluations

## 🎯 Fine-tuning

### Preparing Your Dataset

Organize your audio data in the following structure:

```
data/
├── train/
│   ├── audio1.wav
│   ├── audio1.txt
│   ├── audio2.wav
│   └── audio2.txt
├── eval/
│   ├── audio3.wav
│   └── audio3.txt
└── test/
    ├── audio4.wav
    └── audio4.txt
```

Each `.txt` file should contain the transcription for the corresponding audio file.

### Running Fine-tuning

Using Docker:
```bash
docker-compose up aaci-finetuner
```

Or locally:
```bash
python -m aaci.finetuning
```

### Configuration

Fine-tuning parameters can be adjusted in `.env` or through environment variables:

- `NUM_EPOCHS`: Number of training epochs (default: 10)
- `BATCH_SIZE`: Training batch size (default: 4)
- `LEARNING_RATE`: Learning rate (default: 1e-5)
- `OUTPUT_DIR`: Checkpoint output directory (default: ./checkpoints)

## 🏗️ Architecture

### Components

1. **Worker Container**: FastAPI-based transcription service
2. **Fine-tuner Container**: Training pipeline for model adaptation
3. **Medical Vocabulary**: Portuguese clinical terminology library
4. **Utilities**: Audio processing and validation tools


## 📊 Medical Vocabulary

The system includes extensive Portuguese medical terminology:

- **Psychiatric**: Mental health and psychiatric terminology
- **Abbreviations**: Common clinical abbreviations (PA, FC, AVC, etc.)
- 
## 🛠️ Development

### Project Structure

```
AACI/
├── aaci/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── medical_vocabulary/
│   │   └── __init__.py
│   ├── finetuning/
│   │   └── __init__.py
│   ├── workers/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```


## 🎓 Citation

If you use this system in your research, please cite:

```bibtex
@software{aaci2025,
  title={AACI: Ambient-Agentic Clinical Intelligence},
  author={Voither Health},
  year={2024},
  url={https://github.com/myselfgus/AACI}
}
```
