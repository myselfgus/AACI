"""
Configuration for AACI fine-tuning and inference.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for Whisper model."""
    model_name: str = "openai/whisper-large-v3"
    language: str = "pt"
    task: str = "transcribe"
    device: str = "cuda"
    compute_dtype: str = "float16"
    use_flash_attention: bool = True


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    push_to_hub: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class DataConfig:
    """Configuration for dataset."""
    train_dataset_path: str = "./data/train"
    eval_dataset_path: str = "./data/eval"
    test_dataset_path: str = "./data/test"
    audio_column: str = "audio"
    text_column: str = "transcription"
    max_duration_seconds: float = 30.0
    min_duration_seconds: float = 0.5
    sample_rate: int = 16000


@dataclass
class WorkerConfig:
    """Configuration for worker."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    model_cache_dir: str = "./models"
    max_audio_length: int = 480000  # 30 seconds at 16kHz
    batch_size: int = 8
    

def get_config_from_env():
    """Load configuration from environment variables."""
    model_config = ModelConfig(
        model_name=os.getenv("MODEL_NAME", "openai/whisper-large-v3"),
        language=os.getenv("LANGUAGE", "pt"),
        task=os.getenv("TASK", "transcribe"),
        device=os.getenv("DEVICE", "cuda"),
    )
    
    training_config = TrainingConfig(
        output_dir=os.getenv("OUTPUT_DIR", "./checkpoints"),
        num_train_epochs=int(os.getenv("NUM_EPOCHS", "10")),
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", "4")),
        learning_rate=float(os.getenv("LEARNING_RATE", "1e-5")),
    )
    
    data_config = DataConfig(
        train_dataset_path=os.getenv("TRAIN_DATA", "./data/train"),
        eval_dataset_path=os.getenv("EVAL_DATA", "./data/eval"),
        test_dataset_path=os.getenv("TEST_DATA", "./data/test"),
    )
    
    worker_config = WorkerConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("NUM_WORKERS", "4")),
    )
    
    return model_config, training_config, data_config, worker_config
