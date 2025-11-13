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

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for training and inference)
- Python 3.8+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/myselfgus/AACI.git
cd AACI
```

2. Copy environment configuration:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Build and run with Docker Compose:
```bash
docker-compose up -d aaci-worker
```

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
```

3. Run the worker:
```bash
python -m aaci.workers
```

## 📖 Usage

### Transcription API

The worker exposes a REST API for audio transcription:

```python
import requests

# Transcribe audio file
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f}
    )
    
print(response.json())
# {
#   "text": "Paciente apresenta hipertensão arterial sistêmica...",
#   "language": "pt",
#   "duration": 15.3
# }
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Medical Vocabulary

```bash
curl http://localhost:8000/vocabulary
```

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

### Technology Stack

- **Model**: OpenAI Whisper Large V3
- **Framework**: Hugging Face Transformers
- **API**: FastAPI + Uvicorn
- **Containerization**: Docker + Docker Compose
- **Audio Processing**: Librosa, SoundFile
- **Medical NLP**: spaCy (Portuguese)

## 📊 Medical Vocabulary

The system includes extensive Portuguese medical terminology:

- **General Medical Terms**: Anatomical terms, procedures, diagnoses
- **Cardiovascular**: Cardiology-specific vocabulary
- **Respiratory**: Pulmonology terms
- **Psychiatric**: Mental health and psychiatric terminology
- **Abbreviations**: Common clinical abbreviations (PA, FC, AVC, etc.)

## 🔧 API Reference

### Endpoints

#### POST `/transcribe`
Transcribe an audio file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Audio file (wav, mp3, flac, etc.)

**Response:**
```json
{
  "text": "transcribed text",
  "language": "pt",
  "duration": 15.3,
  "confidence": 0.95
}
```

#### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "openai/whisper-large-v3",
  "device": "cuda"
}
```

#### GET `/vocabulary`
Get medical vocabulary terms.

**Response:**
```json
{
  "terms": ["anamnese", "diagnóstico", ...],
  "count": 150
}
```

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

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black aaci/
flake8 aaci/
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and support, please contact Voither Health.

## 🔒 Security Note

This system handles sensitive medical data. Ensure proper security measures:
- Use secure connections (HTTPS/TLS)
- Implement authentication and authorization
- Comply with HIPAA/LGPD regulations
- Encrypt data at rest and in transit
- Regular security audits

## 🎓 Citation

If you use this system in your research, please cite:

```bibtex
@software{aaci2024,
  title={AACI: Ambient-Agentic Clinical Intelligence},
  author={Voither Health},
  year={2024},
  url={https://github.com/myselfgus/AACI}
}
```
