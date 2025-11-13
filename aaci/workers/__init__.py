"""
AACI Worker for real-time transcription using Whisper Large 3.
"""
import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
from typing import Optional, List
import logging
from pydantic import BaseModel

from ..config import ModelConfig, WorkerConfig, get_config_from_env
from ..medical_vocabulary import expand_abbreviations, get_all_medical_terms


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Response models
class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    language: str
    duration: float
    confidence: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model: str
    device: str


# FastAPI app
app = FastAPI(
    title="AACI Worker API",
    description="Ambient-Agentic Clinical Intelligence Worker for Voither HealthOS",
    version="0.1.0",
)


class WhisperWorker:
    """Worker for Whisper transcription."""
    
    def __init__(self, model_config: ModelConfig, worker_config: WorkerConfig):
        self.model_config = model_config
        self.worker_config = worker_config
        self.processor = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load Whisper model and processor."""
        logger.info(f"Loading model: {self.model_config.model_name}")
        
        try:
            self.processor = WhisperProcessor.from_pretrained(
                self.model_config.model_name,
                cache_dir=self.worker_config.model_cache_dir,
                language=self.model_config.language,
                task=self.model_config.task,
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_config.model_name,
                cache_dir=self.worker_config.model_cache_dir,
                torch_dtype=torch.float16 if self.model_config.compute_dtype == "float16" else torch.float32,
            )
            
            # Move model to device
            if self.model_config.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("Model loaded on CUDA")
            else:
                logger.warning("CUDA not available, using CPU")
                
            self.model.eval()
            
            # Set forced decoder IDs for language and task
            self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=self.model_config.language,
                task=self.model_config.task,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Transcribed text
        """
        try:
            # Resample if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            
            # Process audio
            input_features = self.processor.feature_extractor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            
            # Move to device
            if self.model_config.device == "cuda" and torch.cuda.is_available():
                input_features = input_features.to("cuda")
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # Post-process: expand medical abbreviations
            transcription = expand_abbreviations(transcription)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise


# Global worker instance
worker: Optional[WhisperWorker] = None


@app.on_event("startup")
async def startup_event():
    """Initialize worker on startup."""
    global worker
    model_config, _, _, worker_config = get_config_from_env()
    worker = WhisperWorker(model_config, worker_config)
    logger.info("Worker initialized")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    return HealthResponse(
        status="healthy",
        model=worker.model_config.model_name,
        device=worker.model_config.device,
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file.
    
    Args:
        file: Audio file (wav, mp3, flac, etc.)
        
    Returns:
        Transcription result
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Load audio
        audio_data, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None
        )
        
        # Calculate duration
        duration = len(audio_data) / sample_rate
        
        # Transcribe
        transcription = worker.transcribe(audio_data, sample_rate)
        
        return TranscriptionResponse(
            text=transcription,
            language=worker.model_config.language,
            duration=duration,
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vocabulary")
async def get_medical_vocabulary():
    """Get medical vocabulary terms."""
    return {
        "terms": get_all_medical_terms(),
        "count": len(get_all_medical_terms()),
    }


def main():
    """Run the worker server."""
    import uvicorn
    from ..config import get_config_from_env
    
    _, _, _, worker_config = get_config_from_env()
    
    uvicorn.run(
        "aaci.workers:app",
        host=worker_config.host,
        port=worker_config.port,
        workers=worker_config.workers,
        log_level="info",
    )


if __name__ == "__main__":
    import io
    main()
