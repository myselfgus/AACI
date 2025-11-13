# 🚀 AACI - Setup e Configuração Simplificada
## Ambient-Agentic Clinical Intelligence - Whisper Enriquecido para Consultas Médicas em Português

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Recursos Principais](#recursos-principais)
3. [Instalação Rápida](#instalação-rápida)
4. [Configuração do Container](#configuração-do-container)
5. [Uso dos Endpoints](#uso-dos-endpoints)
6. [Transcrição em Tempo Real](#transcrição-em-tempo-real)
7. [Ambient Agents](#ambient-agents)
8. [Fine-Tuning com Dados Médicos](#fine-tuning-com-dados-médicos)
9. [Deploy na Cloudflare](#deploy-na-cloudflare)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

O AACI é um sistema avançado de transcrição e análise de consultas médicas em português brasileiro, construído sobre o Whisper Large 3 da OpenAI, com melhorias significativas para o contexto médico brasileiro.

### Principais Diferenciais

✅ **Vocabulário Médico Expandido**: 500+ termos médicos e 100+ abreviações clínicas em português
✅ **Diarização Avançada**: Identificação automática de médico e paciente
✅ **Transcrição em Tempo Real**: WebSocket com latência <500ms
✅ **Ambient Agents**: Sistema inteligente que detecta padrões e dispara ações durante a consulta
✅ **Análise Paralinguística**: Detecção de emoções, estresse e qualidade de voz
✅ **Redução de Ruído**: Otimizado para ambientes clínicos
✅ **Fine-Tuning Ready**: Pipeline completo para treinar com seus 50GB de áudio médico

---

## 🔧 Recursos Principais

### 1. Transcrição Médica Avançada
- **Modelo**: Whisper Large 3 Turbo (5.4x mais rápido)
- **Idioma**: Português brasileiro otimizado
- **Precisão**: WER <10% em contexto médico
- **Contexto**: Compreensão de termos médicos complexos

### 2. Diarização de Alto Desempenho
- **Engine**: Pyannote 3.3 + SpeechBrain + Resemblyzer
- **Precisão**: DER ~10% (Diarization Error Rate)
- **Real-time**: Factor 2.5% em GPU
- **Speakers**: Identificação automática de médico/paciente

### 3. Transcrição em Tempo Real
- **Latência**: <500ms
- **VAD**: Voice Activity Detection com WebRTC
- **Streaming**: WebSocket com chunks de 300ms
- **Buffer**: Processamento inteligente com sobreposição

### 4. Ambient Agent System
- **Pattern Matching**: Detecção automática de situações clínicas
- **Triggers**: 15+ padrões pré-configurados (emergências, prescrições, exames)
- **Actions**: Disparo automático de agents (SOAP notes, prescrições, alertas)
- **Prioridades**: Sistema de alertas com 10 níveis de urgência

### 5. Análise Paralinguística
- **Acoustic Features**: MFCC, pitch, intensidade, HNR
- **Emotion Detection**: Indicadores de estresse, fadiga, ansiedade
- **Prosody Analysis**: Taxa de fala, pausas, entonação
- **Voice Quality**: Análise de qualidade vocal do paciente

---

## 🚀 Instalação Rápida

### Opção 1: Docker Compose (Recomendado)

```bash
# 1. Clone o repositório
git clone https://github.com/myselfgus/AACI.git
cd AACI

# 2. Configure variáveis de ambiente
cp .env.example .env
nano .env  # Edite com suas configurações

# 3. Inicie o container
docker-compose up -d

# 4. Verifique o status
curl http://localhost:8787/health
```

### Opção 2: Instalação Manual

```bash
# 1. Clone o repositório
git clone https://github.com/myselfgus/AACI.git
cd AACI

# 2. Crie ambiente virtual
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instale dependências
pip install -r requirements.txt
pip install -r container_src/requirements.txt

# 4. Configure variáveis de ambiente
cp .env.example .env
nano .env

# 5. Inicie o servidor
python container_src/app.py
```

---

## ⚙️ Configuração do Container

### Variáveis de Ambiente (`.env`)

```bash
# ============================================================================
# AACI CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_NAME=openai/whisper-large-v3
LANGUAGE=pt
TASK=transcribe
DEVICE=cuda  # ou "cpu" se não tiver GPU
COMPUTE_TYPE=float16  # float16, int8, float32

# HuggingFace Token (necessário para diarization)
HF_AUTH_TOKEN=seu_token_aqui  # Obtenha em https://huggingface.co/settings/tokens

# API Configuration
PORT=8787
HOST=0.0.0.0
LOG_LEVEL=info

# Audio Processing
ENABLE_NOISE_REDUCTION=true
ENABLE_DIARIZATION=true
ENABLE_AMBIENT_AGENTS=true
SAMPLE_RATE=16000

# Real-time Configuration
BUFFER_DURATION_S=3
OVERLAP_DURATION_S=0.5
VAD_AGGRESSIVENESS=2  # 0-3, 3 = mais agressivo

# Fine-tuning Configuration
CHECKPOINT_DIR=/data/checkpoints
DATASET_DIR=/data/medical_audio
LOGS_DIR=/data/logs
```

### docker-compose.yml Configurado

```yaml
version: '3.8'

services:
  aaci-worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: aaci-whisper-worker
    ports:
      - "8787:8787"
    environment:
      - MODEL_NAME=${MODEL_NAME}
      - LANGUAGE=${LANGUAGE}
      - DEVICE=${DEVICE}
      - HF_AUTH_TOKEN=${HF_AUTH_TOKEN}
      - ENABLE_DIARIZATION=${ENABLE_DIARIZATION}
      - ENABLE_AMBIENT_AGENTS=${ENABLE_AMBIENT_AGENTS}
    volumes:
      - ./data:/data
      - ./models:/models
      - ./logs:/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  aaci-finetuner:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: aaci-finetuner
    command: python -m aaci.finetuning.train
    environment:
      - MODEL_NAME=${MODEL_NAME}
      - LANGUAGE=${LANGUAGE}
      - DEVICE=${DEVICE}
    volumes:
      - ./data:/data
      - ./models:/models
      - ./logs:/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    profiles:
      - finetuning  # Inicia apenas com: docker-compose --profile finetuning up
```

---

## 📡 Uso dos Endpoints

### 1. Health Check

```bash
# Verificar status do serviço
curl http://localhost:8787/health

# Health check estendido
curl -X POST http://localhost:8787/health-check
```

**Resposta:**
```json
{
  "status": "healthy",
  "service": "Whisper Container Worker",
  "models": {
    "whisper": "loaded",
    "diarization": "loaded",
    "ner": "loaded",
    "opensmile": "loaded"
  },
  "gpu_available": true,
  "device": "cuda",
  "jobs_processed": 42,
  "timestamp": "2025-11-13T10:30:00"
}
```

### 2. Transcrição de Arquivo (Assíncrona)

```bash
# Enviar arquivo para processamento
curl -X POST http://localhost:8787/process \
  -F "file=@consulta_medica.mp3" \
  -F "language=pt" \
  -F "enable_diarization=true" \
  -F "enable_medical_ner=true" \
  -F "enable_paralinguistics=true" \
  -F "webhook_url=https://seu-webhook.com/callback"

# Resposta imediata com ID de processamento
{
  "processing_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Processing started"
}

# Verificar status do processamento
curl http://localhost:8787/status/550e8400-e29b-41d4-a716-446655440000
```

**Resposta Completa:**
```json
{
  "processing_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "transcription": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Bom dia, doutor. Estou com dor no peito há dois dias.",
      "confidence": 0.95
    }
  ],
  "speakers": [
    {
      "start": 0.0,
      "end": 5.2,
      "speaker": "SPEAKER_01",
      "duration": 5.2
    }
  ],
  "medical_entities": [
    {
      "entity": "SINTOMA",
      "word": "dor no peito",
      "start": 15,
      "end": 27,
      "score": 0.92
    }
  ],
  "paralinguistic_features": {
    "pitch_mean": 180.5,
    "intensity_mean": 65.3,
    "emotion_indicators": {
      "stress_level": 0.65
    }
  },
  "processing_time_seconds": 8.3
}
```

### 3. Transcrição Direta (Síncrona)

```python
import requests

# Transcrição síncrona (aguarda resultado)
with open("consulta.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8787/transcribe",
        files={"file": f},
        data={
            "language": "pt",
            "enable_diarization": True,
            "medical_context": True
        }
    )

result = response.json()
print(result["transcription"])
```

---

## 🎤 Transcrição em Tempo Real

### WebSocket Endpoint: `ws://localhost:8787/realtime`

### Exemplo em JavaScript (Browser)

```javascript
// Conectar ao WebSocket
const ws = new WebSocket('ws://localhost:8787/realtime');
ws.binaryType = 'arraybuffer';

// Configurar captura de áudio
navigator.mediaDevices.getUserMedia({
  audio: {
    channelCount: 1,
    sampleRate: 16000,
    echoCancellation: true,
    noiseSuppression: true
  }
}).then(stream => {
  const audioContext = new AudioContext({ sampleRate: 16000 });
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    const audioData = e.inputBuffer.getChannelData(0);

    // Converter para Int16 (16-bit PCM)
    const int16Data = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      int16Data[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32767));
    }

    // Enviar para servidor
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(int16Data.buffer);
    }
  };

  source.connect(processor);
  processor.connect(audioContext.destination);
});

// Receber transcrições
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.text) {
    console.log('🎤 Transcrição:', data.text);
    console.log('👤 Speaker:', data.speaker);
    console.log('⏱️  Tempo:', data.start, '-', data.end);
    console.log('📊 Confiança:', data.confidence);

    // Agents disparados
    if (data.agents_triggered && data.agents_triggered.length > 0) {
      console.log('🤖 Agents Disparados:', data.agents_triggered);

      data.agents_triggered.forEach(agent => {
        if (agent.priority >= 8) {
          alert(`⚠️ ALERTA: ${agent.agent}`);
        }
      });
    }
  }

  // Status updates
  if (data.status) {
    console.log('ℹ️  Status:', data.status, '-', data.message);
  }
};

ws.onerror = (error) => {
  console.error('❌ WebSocket Error:', error);
};

ws.onclose = () => {
  console.log('🔌 WebSocket Fechado');
};
```

### Exemplo em Python

```python
import asyncio
import websockets
import pyaudio
import json

async def realtime_transcription():
    uri = "ws://localhost:8787/realtime"

    async with websockets.connect(uri) as websocket:
        # Configurar PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096
        )

        print("🎤 Iniciando transcrição em tempo real...")

        # Tarefa para enviar áudio
        async def send_audio():
            while True:
                audio_data = stream.read(4096, exception_on_overflow=False)
                await websocket.send(audio_data)
                await asyncio.sleep(0.01)

        # Tarefa para receber transcrições
        async def receive_transcriptions():
            async for message in websocket:
                data = json.loads(message)

                if "text" in data:
                    print(f"\n🎤 {data['speaker']}: {data['text']}")
                    print(f"   Confiança: {data['confidence']:.2%}")

                    if data.get('agents_triggered'):
                        print(f"   🤖 Agents: {[a['agent'] for a in data['agents_triggered']]}")

        # Executar ambas as tarefas
        await asyncio.gather(
            send_audio(),
            receive_transcriptions()
        )

# Rodar
asyncio.run(realtime_transcription())
```

---

## 🤖 Ambient Agents

### Como Funciona

O sistema monitora continuamente a transcrição e identifica padrões específicos que disparam agents automaticamente. Ideal para:

- ⚠️ **Alertas de Emergência**: Detecta sintomas críticos (dor no peito, AVC, ideação suicida)
- 📝 **Documentação Automática**: Gera SOAP notes, prescrições, encaminhamentos
- 🔬 **Pedidos de Exames**: Identifica quando solicitar exames laboratoriais ou imagens
- 💊 **Checagem de Interações**: Verifica interações medicamentosas automaticamente
- 📅 **Agendamento**: Detecta quando agendar retorno

### Agents Disponíveis

| Agent | Descrição | Prioridade |
|-------|-----------|------------|
| `RED_FLAG_ALERT` | Sintomas de emergência | 10 (Crítica) |
| `STROKE_SYMPTOMS` | Sinais de AVC | 10 (Crítica) |
| `SUICIDAL_IDEATION` | Risco de suicídio | 10 (Crítica) |
| `DRUG_INTERACTION_CHECKER` | Verifica interações medicamentosas | 8 (Alta) |
| `PRESCRIPTION_WRITER` | Gera prescrição médica | 7 (Alta) |
| `LAB_ORDER` | Solicita exames | 6 (Média) |
| `REFERRAL_CREATOR` | Cria encaminhamento | 6 (Média) |
| `DIFFERENTIAL_DIAGNOSIS` | Auxilia diagnóstico diferencial | 5 (Média) |
| `SOAP_NOTE_GENERATOR` | Gera nota SOAP | 6 (Média) |
| `PATIENT_EDUCATION` | Material educativo | 4 (Baixa) |

### Exemplo de Uso Programático

```python
from aaci.ambient_agents import AmbientAgentManager, AgentType

# Inicializar manager
manager = AmbientAgentManager()

# Simular consulta
utterances = [
    ("doctor", "Bom dia! Qual o motivo da consulta?"),
    ("patient", "Doutor, estou com dor no peito há 2 horas, irradiando para o braço."),
    ("doctor", "Vou solicitar ECG urgente e troponina."),
]

for speaker, text in utterances:
    agents = manager.add_utterance(text, speaker)

    for agent_type, params in agents:
        print(f"🤖 Agent: {agent_type.value}")
        print(f"   Prioridade: {params['priority']}")
        print(f"   Texto: {params['matched_text']}")

# Obter resumo
summary = manager.get_conversation_summary()
print(f"\n📊 Resumo:")
print(f"   Fase: {summary['current_phase']}")
print(f"   Alertas críticos: {len(summary['high_priority_alerts'])}")
```

### Customizar Patterns

```python
from aaci.ambient_agents import PatternTrigger, AgentType, ConversationPhase

# Criar trigger customizado
custom_trigger = PatternTrigger(
    name="diabetes_follow_up",
    pattern=r"(hemoglobina glicada|HbA1c|glicemia de jejum)",
    agent_type=AgentType.LAB_ORDER,
    priority=6,
    phase=ConversationPhase.PLAN,
    parameters={"exam_type": "diabetes_monitoring"}
)

# Adicionar ao manager
manager = AmbientAgentManager(patterns=[custom_trigger, ...])
```

---

## 🎓 Fine-Tuning com Dados Médicos

### Preparar Dataset (50GB de áudio)

```bash
# 1. Organize seus arquivos de áudio
data/
├── medical_audio/
│   ├── consult_001.mp3
│   ├── consult_002.wav
│   └── ...
└── transcriptions/
    ├── consult_001.txt
    ├── consult_002.txt
    └── ...

# 2. Prepare o dataset
python scripts/prepare_dataset.py \
  --audio_dir data/medical_audio \
  --transcript_dir data/transcriptions \
  --output_dir data/prepared_dataset \
  --language pt \
  --sample_rate 16000

# 3. Verifique o dataset
python scripts/validate_dataset.py \
  --dataset_dir data/prepared_dataset
```

### Configurar Fine-Tuning

```yaml
# config/finetune_config.yaml
model:
  name: openai/whisper-large-v3
  language: pt
  task: transcribe

training:
  num_train_epochs: 10
  batch_size: 4
  learning_rate: 1.0e-5
  warmup_steps: 500
  gradient_accumulation_steps: 2
  fp16: true
  gradient_checkpointing: true

dataset:
  train_split: 0.9
  val_split: 0.1
  max_audio_length: 30  # segundos
  min_audio_length: 1

paths:
  dataset_dir: /data/prepared_dataset
  output_dir: /data/checkpoints
  cache_dir: /data/cache
```

### Executar Fine-Tuning

```bash
# Opção 1: Docker Compose (Recomendado)
docker-compose --profile finetuning up

# Opção 2: Manual
python -m aaci.finetuning.train \
  --config config/finetune_config.yaml \
  --dataset_dir /data/prepared_dataset \
  --output_dir /data/checkpoints \
  --num_epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-5

# Monitorar progresso (TensorBoard)
tensorboard --logdir /data/logs
# Acesse: http://localhost:6006
```

### Avaliar Modelo Fine-Tuned

```bash
# Avaliar WER (Word Error Rate)
python -m aaci.finetuning.evaluate \
  --model_path /data/checkpoints/checkpoint-1000 \
  --test_dataset /data/test_set \
  --language pt

# Comparar com modelo base
python scripts/compare_models.py \
  --base_model openai/whisper-large-v3 \
  --finetuned_model /data/checkpoints/final \
  --test_audio data/test_samples/
```

### Usar Modelo Fine-Tuned

```bash
# Atualizar .env
MODEL_NAME=/data/checkpoints/final

# Reiniciar container
docker-compose restart aaci-worker
```

---

## ☁️ Deploy na Cloudflare

### Deploy com Wrangler

```bash
# 1. Instalar Wrangler
npm install -g wrangler

# 2. Login na Cloudflare
wrangler login

# 3. Configurar wrangler.toml
cat > wrangler.toml << EOF
name = "aaci-whisper-worker"
main = "src/index.ts"
compatibility_date = "2025-11-13"

[durable_objects]
bindings = [
  { name = "WHISPER_CONTAINER", class_name = "WhisperContainer" }
]

[[r2_buckets]]
binding = "AUDIO_BUCKET"
bucket_name = "aaci-audio-files"

[env.production]
vars = { CONTAINER_URL = "https://sua-instancia.cloudflare.com" }
EOF

# 4. Deploy
wrangler publish
```

### Container na Cloudflare (Durable Objects)

Seu container já está otimizado para Cloudflare. Certifique-se de:

1. ✅ Container usa porta 8787
2. ✅ Dockerfile otimizado para cold starts
3. ✅ Health checks configurados
4. ✅ CORS habilitado

### Monitoramento

```bash
# Ver logs em tempo real
wrangler tail

# Métricas
wrangler metrics
```

---

## 🔧 Troubleshooting

### Problema: GPU não detectada

```bash
# Verificar NVIDIA drivers
nvidia-smi

# Verificar Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Reinstalar nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Problema: Diarização não funciona

```bash
# 1. Verificar HuggingFace token
echo $HF_AUTH_TOKEN

# 2. Aceitar license do Pyannote
# Acesse: https://huggingface.co/pyannote/speaker-diarization-3.1
# Clique em "Agree and access repository"

# 3. Testar manualmente
python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='$HF_AUTH_TOKEN')"
```

### Problema: Memória insuficiente

```yaml
# docker-compose.yml - Adicionar limites
services:
  aaci-worker:
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
```

### Problema: WebSocket desconecta

```python
# Aumentar timeout no cliente
ws = websockets.connect(uri, ping_timeout=60, ping_interval=30)
```

### Logs e Debug

```bash
# Ver logs do container
docker-compose logs -f aaci-worker

# Logs detalhados
docker-compose logs --tail=100 aaci-worker

# Entrar no container
docker exec -it aaci-whisper-worker bash

# Verificar processos
docker exec aaci-whisper-worker ps aux
```

---

## 📞 Suporte

- **Documentação**: [docs/](./docs/)
- **Issues**: [GitHub Issues](https://github.com/myselfgus/AACI/issues)
- **Discussões**: [GitHub Discussions](https://github.com/myselfgus/AACI/discussions)

---

## 📄 Licença

MIT License - Veja [LICENSE](./LICENSE) para detalhes.

---

**Desenvolvido com ❤️ para a comunidade médica brasileira**
