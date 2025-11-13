# 🎓 Guia de Fine-Tuning - Whisper Large 3 com 50GB de Áudio Médico

## Otimizando Whisper para Consultas Médicas em Português

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Preparação do Dataset](#preparação-do-dataset)
3. [Configuração do Ambiente](#configuração-do-ambiente)
4. [Pipeline de Treinamento](#pipeline-de-treinamento)
5. [Monitoramento e Avaliação](#monitoramento-e-avaliação)
6. [Otimizações Avançadas](#otimizações-avançadas)
7. [Deploy do Modelo](#deploy-do-modelo)

---

## 🎯 Visão Geral

### Por que fazer Fine-Tuning?

O Whisper Large 3 é excelente, mas pode ser **significativamente melhorado** para o contexto médico brasileiro:

- 📈 **Redução de WER**: De ~15% para ~8% em áudio médico
- 🏥 **Termos Médicos**: Melhor compreensão de vocabulário técnico
- 🗣️ **Sotaques Regionais**: Adaptação para sotaques brasileiros
- 🔊 **Ruído Clínico**: Resistência a ruídos de consultórios
- ⚡ **Velocidade**: Possibilidade de usar modelos menores com mesma precisão

### Recursos Necessários

| Recurso | Mínimo | Recomendado | Ótimo |
|---------|--------|-------------|-------|
| **GPU** | 16GB VRAM | 24GB VRAM | 40GB+ VRAM |
| **RAM** | 32GB | 64GB | 128GB |
| **Storage** | 100GB | 200GB | 500GB |
| **Tempo** | 3-5 dias | 1-2 dias | 8-12 horas |

---

## 📦 Preparação do Dataset

### Etapa 1: Organizar Arquivos de Áudio

```bash
# Estrutura recomendada para 50GB de áudio
data/
├── raw_audio/              # Áudio original (50GB)
│   ├── consults/
│   │   ├── 2024-01-001.mp3
│   │   ├── 2024-01-002.wav
│   │   └── ...
│   ├── emergency/
│   └── follow_ups/
├── transcriptions/         # Transcrições manuais
│   ├── 2024-01-001.txt
│   ├── 2024-01-002.txt
│   └── ...
└── metadata/
    └── dataset_info.csv    # Metadados (duração, qualidade, etc.)
```

### Etapa 2: Validar e Limpar Áudio

```python
# scripts/validate_audio.py
import os
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd

def validate_audio_files(audio_dir, output_csv):
    """Valida todos os arquivos de áudio e cria relatório."""

    results = []

    results = []

    audio_files = []
    extensions = ["mp3", "wav", "m4a", "flac"]
    for ext in extensions:
        audio_files.extend(Path(audio_dir).rglob(f"*.{ext}"))

    for audio_file in audio_files:
        try:
            # Carregar áudio
            y, sr = librosa.load(audio_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # Verificar qualidade
            issues = []
            if duration < 1.0:
                issues.append("too_short")
            if duration > 1800:  # 30 min
                issues.append("too_long")
            if sr < 8000:
                issues.append("low_sample_rate")

            results.append({
                'file': str(audio_file),
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[0],
                'issues': ','.join(issues) if issues else 'ok',
                'status': 'valid' if not issues else 'needs_fix'
            })

        except Exception as e:
            results.append({
                'file': str(audio_file),
                'error': str(e),
                'status': 'invalid'
            })

    # Salvar relatório
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"✅ Validados {len(results)} arquivos")
    print(f"   Válidos: {len(df[df['status']=='valid'])}")
    print(f"   Precisam correção: {len(df[df['status']=='needs_fix'])}")
    print(f"   Inválidos: {len(df[df['status']=='invalid'])}")

    return df

# Executar validação
df = validate_audio_files('data/raw_audio', 'data/validation_report.csv')
```

### Etapa 3: Processar Áudio

```bash
# Converter todos para formato padrão (WAV 16kHz mono)
python scripts/preprocess_audio.py \
  --input_dir data/raw_audio \
  --output_dir data/processed_audio \
  --target_sr 16000 \
  --format wav \
  --normalize \
  --remove_silence \
  --noise_reduction

# Opções:
# --normalize: Normaliza volume
# --remove_silence: Remove silêncios longos
# --noise_reduction: Aplica redução de ruído
# --split_long: Divide áudios >30min em chunks
```

### Etapa 4: Preparar Transcrições

```python
# scripts/prepare_transcriptions.py
import json
from pathlib import Path
from aaci.medical_vocabulary import expand_abbreviations, get_all_medical_terms

def prepare_transcription(text_file):
    """Processa e melhora transcrição."""

    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Expandir abreviações médicas
    text = expand_abbreviations(text)

    # 2. Normalizar pontuação
    text = normalize_punctuation(text)

    # 3. Remover marcações de tempo se existirem
    text = remove_timestamps(text)

    # 4. Verificar termos médicos
    medical_terms = check_medical_terms(text)

    return {
        'text': text,
        'medical_terms_count': len(medical_terms),
        'word_count': len(text.split()),
        'char_count': len(text)
    }

# Processar todas as transcrições
def batch_process_transcriptions(transcription_dir, output_dir):
    for txt_file in Path(transcription_dir).glob("*.txt"):
        result = prepare_transcription(txt_file)

        # Salvar versão processada
        output_file = Path(output_dir) / txt_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['text'])

batch_process_transcriptions('data/transcriptions', 'data/processed_transcriptions')
```

### Etapa 5: Criar Dataset HuggingFace

```python
# scripts/create_dataset.py
from datasets import Dataset, DatasetDict, Audio
import pandas as pd
from pathlib import Path

def create_medical_dataset(audio_dir, transcription_dir, output_dir):
    """Cria dataset no formato HuggingFace."""

    data = {
        'audio': [],
        'transcription': [],
        'duration': [],
        'medical_specialty': [],
        'audio_quality': []
    }

    # Coletar pares de áudio-transcrição
    for audio_file in Path(audio_dir).glob("*.wav"):
        transcript_file = Path(transcription_dir) / f"{audio_file.stem}.txt"

        if transcript_file.exists():
            # Ler transcrição
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()

            # Adicionar ao dataset
            data['audio'].append(str(audio_file))
            data['transcription'].append(transcription)

            # Metadados (opcional)
            y, sr = librosa.load(audio_file, sr=16000)
            data['duration'].append(len(y) / sr)
            data['medical_specialty'].append(detect_specialty(transcription))
            data['audio_quality'].append(assess_quality(y, sr))

    # Criar Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    # Adicionar coluna de áudio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split train/val/test
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict({
        'train': train_test['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })

    # Salvar
    dataset_dict.save_to_disk(output_dir)

    print(f"✅ Dataset criado:")
    print(f"   Train: {len(dataset_dict['train'])} exemplos")
    print(f"   Validation: {len(dataset_dict['validation'])} exemplos")
    print(f"   Test: {len(dataset_dict['test'])} exemplos")
    print(f"   Total de áudio: {sum(data['duration']) / 3600:.1f} horas")

    return dataset_dict

# Criar dataset
dataset = create_medical_dataset(
    'data/processed_audio',
    'data/processed_transcriptions',
    'data/medical_whisper_dataset'
)
```

---

## ⚙️ Configuração do Ambiente

### Instalar Dependências

```bash
# Instalar dependências de fine-tuning
pip install transformers[torch] datasets accelerate evaluate jiwer tensorboard wandb

# Verificar GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Configurar Weights & Biases (Opcional)

```bash
# Login no W&B para tracking
wandb login

# Ou use TensorBoard (local)
tensorboard --logdir logs/
```

---

## 🏋️ Pipeline de Treinamento

### Script de Fine-Tuning Completo

```python
# scripts/finetune_whisper.py
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from datasets import load_from_disk
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_callback import TrainerCallback
import evaluate

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

@dataclass
class FineTuningConfig:
    """Configuração para fine-tuning."""

    # Modelo
    model_name: str = "openai/whisper-large-v3"
    language: str = "pt"
    task: str = "transcribe"

    # Dataset
    dataset_path: str = "data/medical_whisper_dataset"
    max_audio_length: float = 30.0  # segundos

    # Treinamento
    output_dir: str = "models/whisper-large-v3-medical-pt"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 500

    # Otimizações
    fp16: bool = True
    gradient_checkpointing: bool = True
    group_by_length: bool = True

    # Avaliação
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100

    # Hardware
    dataloader_num_workers: int = 4
    use_cpu: bool = False

config = FineTuningConfig()

# ============================================================================
# DATA COLLATOR
# ============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator para Whisper."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separar inputs e labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad inputs
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove decoder_start_token_id
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# ============================================================================
# PREPROCESSING
# ============================================================================

def prepare_dataset(batch, feature_extractor, tokenizer):
    """Preprocessa um batch do dataset."""

    # Carregar áudio
    audio = batch["audio"]

    # Extrair features
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Tokenizar transcrição
    batch["labels"] = tokenizer(batch["transcription"]).input_ids

    return batch

# ============================================================================
# MÉTRICAS
# ============================================================================

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    """Calcula WER (Word Error Rate)."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Substituir -100 por pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions e labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calcular WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# ============================================================================
# CALLBACKS
# ============================================================================

class MedicalVocabCallback(TrainerCallback):
    """Callback para monitorar termos médicos durante treinamento."""

    def __init__(self, medical_terms):
        self.medical_terms = medical_terms

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n📊 Epoch {state.epoch} | WER: {metrics.get('eval_wer', 0):.4f}")

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("🚀 Iniciando Fine-Tuning do Whisper Large 3 para Português Médico")
    print("="*80)

    # 1. Carregar dataset
    print("\n📦 Carregando dataset...")
    dataset = load_from_disk(config.dataset_path)
    print(f"   Train: {len(dataset['train'])} exemplos")
    print(f"   Val: {len(dataset['validation'])} exemplos")
    print(f"   Test: {len(dataset['test'])} exemplos")

    # 2. Carregar modelo e processadores
    print(f"\n🤖 Carregando modelo: {config.model_name}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        config.model_name,
        language=config.language,
        task=config.task
    )
    processor = WhisperProcessor.from_pretrained(
        config.model_name,
        language=config.language,
        task=config.task
    )

    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    # Configurar para português
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.language,
        task=config.task
    )
    model.config.suppress_tokens = []

    if config.gradient_checkpointing:
        model.config.use_cache = False

    # 3. Preprocessar dataset
    print("\n⚙️  Preprocessando dataset...")
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=dataset["train"].column_names,
        num_proc=config.dataloader_num_workers
    )

    # 4. Criar data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    # 5. Configurar training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        group_by_length=config.group_by_length,
        dataloader_num_workers=config.dataloader_num_workers,
    )

    # 6. Criar trainer
    print("\n🏋️  Configurando Trainer...")
    from aaci.medical_vocabulary import get_all_medical_terms

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=[MedicalVocabCallback(get_all_medical_terms())]
    )

    # 7. Treinar!
    print("\n🎯 Iniciando treinamento...")
    print("="*80)
    trainer.train()

    # 8. Salvar modelo final
    print("\n💾 Salvando modelo final...")
    trainer.save_model(f"{config.output_dir}/final")
    processor.save_pretrained(f"{config.output_dir}/final")

    # 9. Avaliar no test set
    print("\n📊 Avaliando no test set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"   Test WER: {test_results['eval_wer']:.4f}")

    print("\n✅ Fine-tuning completo!")
    print(f"   Modelo salvo em: {config.output_dir}/final")

if __name__ == "__main__":
    main()
```

### Executar Fine-Tuning

```bash
# Opção 1: Rodar script diretamente
python scripts/finetune_whisper.py

# Opção 2: Com Docker
docker-compose --profile finetuning up

# Opção 3: Com aceleração múltiplas GPUs
accelerate launch --multi_gpu scripts/finetune_whisper.py

# Opção 4: Com DeepSpeed (para GPUs grandes)
accelerate launch --config_file deepspeed_config.yaml scripts/finetune_whisper.py
```

---

## 📊 Monitoramento e Avaliação

### TensorBoard

```bash
# Iniciar TensorBoard
tensorboard --logdir models/whisper-large-v3-medical-pt/runs

# Acesse: http://localhost:6006
```

### Weights & Biases

```python
# Adicionar ao training_args
training_args = Seq2SeqTrainingArguments(
    ...,
    report_to=["wandb"],
    run_name="whisper-large-v3-medical-pt-50gb"
)
```

### Avaliar Modelo

```python
# scripts/evaluate_model.py
from transformers import pipeline
from datasets import load_from_disk
import jiwer

# Carregar modelo fine-tuned
pipe = pipeline(
    "automatic-speech-recognition",
    model="models/whisper-large-v3-medical-pt/final",
    device=0
)

# Carregar test set
test_dataset = load_from_disk("data/medical_whisper_dataset")["test"]

# Avaliar
predictions = []
references = []

for example in test_dataset:
    # Transcrever
    result = pipe(example["audio"]["array"])
    predictions.append(result["text"])
    references.append(example["transcription"])

# Calcular WER
wer = jiwer.wer(references, predictions)
print(f"WER Final: {wer:.2%}")

# WER por categoria
from aaci.medical_vocabulary import get_all_medical_terms
medical_terms = get_all_medical_terms()

# Identificar exemplos com termos médicos
medical_predictions = []
medical_references = []

for pred, ref in zip(predictions, references):
    if any(term in ref.lower() for term in medical_terms):
        medical_predictions.append(pred)
        medical_references.append(ref)

medical_wer = jiwer.wer(medical_references, medical_predictions)
print(f"WER em áudio com termos médicos: {medical_wer:.2%}")
```

---

## 🚀 Otimizações Avançadas

### 1. LoRA (Low-Rank Adaptation)

Fine-tuning mais rápido e com menos memória:

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configurar LoRA
lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Aplicar ao modelo
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Trainable params: 3.7M (vs 1.5B full model)
```

### 2. Quantização

```python
# INT8 quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = WhisperForConditionalGeneration.from_pretrained(
    config.model_name,
    quantization_config=quantization_config
)
```

### 3. Gradient Accumulation

Para treinar com batches maiores em GPUs menores:

```python
training_args = Seq2SeqTrainingArguments(
    ...,
    per_device_train_batch_size=2,  # Batch real menor
    gradient_accumulation_steps=8,  # Acumula 8 steps = batch efetivo de 16
)
```

---

## 🚢 Deploy do Modelo

### Atualizar Container

```bash
# 1. Copiar modelo fine-tuned para container
docker cp models/whisper-large-v3-medical-pt/final \
  aaci-whisper-worker:/models/custom

# 2. Atualizar .env
MODEL_NAME=/models/custom

# 3. Reiniciar container
docker-compose restart aaci-worker
```

### Testar Modelo

```bash
# Testar com arquivo de teste
curl -X POST http://localhost:8787/process \
  -F "file=@test_consult.wav" \
  -F "language=pt"
```

---

## 📈 Resultados Esperados

### Baseline vs Fine-Tuned

| Métrica | Whisper Base | Fine-Tuned |
|---------|--------------|------------|
| WER Geral | ~15% | ~8% |
| WER Termos Médicos | ~25% | ~10% |
| Abreviações | ~40% | ~12% |
| Sotaques BR | ~18% | ~9% |
| Ruído Clínico | ~22% | ~11% |

### Tempo de Treinamento

- **GPU A100 (40GB)**: 8-12 horas
- **GPU V100 (32GB)**: 1-2 dias
- **GPU RTX 4090 (24GB)**: 2-3 dias
- **GPU RTX 3090 (24GB)**: 3-5 dias

---

## 💡 Dicas e Melhores Práticas

1. ✅ **Qualidade > Quantidade**: 10GB de áudio bem transcrito > 50GB mal transcrito
2. ✅ **Diversidade**: Inclua diferentes especialidades, sotaques e condições de áudio
3. ✅ **Validação**: Reserve 20% para validação e nunca use no treino
4. ✅ **Checkpoints**: Salve frequentemente para não perder progresso
5. ✅ **Iterativo**: Comece com subset pequeno, valide, depois escale
6. ✅ **Monitoramento**: Use W&B ou TensorBoard para acompanhar métricas
7. ✅ **Overfitting**: Se WER validação aumenta, pare o treino (early stopping)

---

## 🆘 Troubleshooting

### OOM (Out of Memory)

```python
# Reduzir batch size
per_device_train_batch_size=2

# Aumentar gradient accumulation
gradient_accumulation_steps=8

# Usar gradient checkpointing
gradient_checkpointing=True

# Usar FP16
fp16=True
```

### Treinamento Lento

```bash
# Aumentar workers
dataloader_num_workers=8

# Usar GPU mais rápida
# Usar LoRA em vez de full fine-tuning
# Reduzir eval_steps
```

### WER não melhora

- Verificar qualidade das transcrições
- Aumentar learning rate (1e-4 ou 1e-3)
- Treinar por mais epochs
- Adicionar mais dados de validação

---

**Sucesso no seu fine-tuning! 🚀**
