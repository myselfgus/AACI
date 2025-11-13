"""
Whisper Transcription Server for HealthOS
Handles audio transcription with speaker diarization for medical consultations
Optimized for Portuguese (Brazil)
"""

import os
import sys
import time
import json
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import whisper
import torch
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Optional: Advanced diarization
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("Warning: pyannote.audio not available. Diarization will be limited.")

# Configuration
PORT = int(os.getenv("PORT", "9998"))
MODEL_NAME = os.getenv("MODEL_NAME", "large-v3")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "pt")
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "mp3,m4a,wav,flac,ogg").split(",")

# Initialize FastAPI
app = FastAPI(
    title="Whisper Transcription Server",
    description="Medical audio transcription with speaker diarization for HealthOS",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
whisper_model = None
diarization_pipeline = None


def load_whisper_model():
    """Load Whisper model into memory"""
    global whisper_model

    if whisper_model is None:
        print(f"Loading Whisper model: {MODEL_NAME}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        whisper_model = whisper.load_model(MODEL_NAME, device=device)
        print("Whisper model loaded successfully!")

    return whisper_model


def load_diarization_pipeline():
    """Load speaker diarization pipeline"""
    global diarization_pipeline

    if not DIARIZATION_AVAILABLE:
        return None

    if diarization_pipeline is None and ENABLE_DIARIZATION:
        print("Loading diarization pipeline...")
        try:
            # Note: Requires HuggingFace token for pyannote models
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                print("Diarization pipeline loaded successfully!")
            else:
                print("Warning: No HUGGINGFACE_TOKEN found. Diarization disabled.")
        except Exception as e:
            print(f"Error loading diarization pipeline: {e}")
            return None

    return diarization_pipeline


def simple_diarization(audio_path: str, segments: List[Dict]) -> List[Dict]:
    """
    Simple speaker diarization based on audio features
    This is a basic implementation - for production use pyannote.audio
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)

        # Extract MFCC features for each segment
        enhanced_segments = []

        for segment in segments:
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)

            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) > 0:
                # Extract basic features
                mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)

                # Simple speaker assignment based on pitch
                pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

                # Assign speaker based on pitch range (basic heuristic)
                if pitch_mean > 200:
                    speaker = "SPEAKER_1"  # Higher pitch
                elif pitch_mean > 150:
                    speaker = "SPEAKER_2"  # Medium pitch
                else:
                    speaker = "SPEAKER_3"  # Lower pitch

                segment["speaker"] = speaker
                segment["confidence"] = 0.6  # Low confidence for simple method
            else:
                segment["speaker"] = "UNKNOWN"
                segment["confidence"] = 0.0

            enhanced_segments.append(segment)

        return enhanced_segments

    except Exception as e:
        print(f"Error in simple diarization: {e}")
        # Return original segments with unknown speaker
        for segment in segments:
            segment["speaker"] = "UNKNOWN"
            segment["confidence"] = 0.0
        return segments


def advanced_diarization(audio_path: str, segments: List[Dict]) -> List[Dict]:
    """
    Advanced speaker diarization using pyannote.audio
    """
    try:
        pipeline = load_diarization_pipeline()
        if pipeline is None:
            return simple_diarization(audio_path, segments)

        # Run diarization
        diarization = pipeline(audio_path)

        # Map speakers to segments
        enhanced_segments = []

        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            mid_time = (start_time + end_time) / 2

            # Find speaker at the middle of the segment
            speaker = "UNKNOWN"
            confidence = 0.0

            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if turn.start <= mid_time <= turn.end:
                    speaker = speaker_label
                    confidence = 0.9
                    break

            segment["speaker"] = speaker
            segment["confidence"] = confidence
            enhanced_segments.append(segment)

        return enhanced_segments

    except Exception as e:
        print(f"Error in advanced diarization: {e}")
        return simple_diarization(audio_path, segments)


def transcribe_audio(
    audio_path: str,
    language: str = "pt",
    enable_diarization: bool = True,
    medical_context: bool = True
) -> Dict[str, Any]:
    """
    Transcribe audio file with optional speaker diarization
    """
    try:
        model = load_whisper_model()

        print(f"Transcribing audio: {audio_path}")
        print(f"Language: {language}, Diarization: {enable_diarization}")

        # Transcribe with Whisper
        start_time = time.time()

        # Configure transcription options
        options = {
            "language": language,
            "task": "transcribe",
            "verbose": False,
            "temperature": 0.0,  # More deterministic for medical accuracy
        }

        # Add medical vocabulary hints if needed
        if medical_context:
            # Whisper doesn't support custom vocabulary, but we can post-process
            options["initial_prompt"] = (
                "Esta é uma consulta médica em português do Brasil. "
                "O áudio contém termos médicos e conversação entre médico e paciente."
            )

        result = model.transcribe(audio_path, **options)

        transcription_time = time.time() - start_time
        print(f"Transcription completed in {transcription_time:.2f}s")

        # Extract segments
        segments = [
            {
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "tokens": seg.get("tokens", []),
                "temperature": seg.get("temperature", 0.0),
                "avg_logprob": seg.get("avg_logprob", 0.0),
                "compression_ratio": seg.get("compression_ratio", 0.0),
                "no_speech_prob": seg.get("no_speech_prob", 0.0)
            }
            for i, seg in enumerate(result["segments"])
        ]

        # Perform diarization if enabled
        if enable_diarization and segments:
            print("Performing speaker diarization...")
            diarization_start = time.time()

            if DIARIZATION_AVAILABLE and diarization_pipeline:
                segments = advanced_diarization(audio_path, segments)
            else:
                segments = simple_diarization(audio_path, segments)

            diarization_time = time.time() - diarization_start
            print(f"Diarization completed in {diarization_time:.2f}s")

        # Build response
        response = {
            "success": True,
            "transcript": result["text"].strip(),
            "language": result.get("language", language),
            "segments": segments,
            "metadata": {
                "model": MODEL_NAME,
                "duration": result.get("duration", 0.0),
                "transcription_time": transcription_time,
                "diarization_enabled": enable_diarization,
                "diarization_method": "advanced" if DIARIZATION_AVAILABLE else "simple",
                "medical_context": medical_context,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        return response

    except Exception as e:
        print(f"Error in transcription: {e}")
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Whisper Transcription Server",
        "version": "1.0.0",
        "status": "running",
        "model": MODEL_NAME,
        "language": DEFAULT_LANGUAGE,
        "diarization": ENABLE_DIARIZATION,
        "diarization_available": DIARIZATION_AVAILABLE
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded or can be loaded
        model = load_whisper_model()

        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "diarization_available": DIARIZATION_AVAILABLE,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(DEFAULT_LANGUAGE),
    enable_diarization: Optional[bool] = Form(True),
    medical_context: Optional[bool] = Form(True)
):
    """
    Transcribe audio file with optional speaker diarization

    Parameters:
    - file: Audio file (mp3, m4a, wav, flac, ogg)
    - language: Language code (default: pt for Portuguese)
    - enable_diarization: Enable speaker diarization (default: True)
    - medical_context: Optimize for medical context (default: True)
    """
    temp_path = None

    try:
        # Validate file format
        file_ext = Path(file.filename).suffix.lower().lstrip(".")
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size_mb:.2f}MB. Max: {MAX_FILE_SIZE_MB}MB"
            )

        print(f"Received file: {file.filename} ({file_size_mb:.2f}MB)")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        print(f"Saved to: {temp_path}")

        # Transcribe
        result = transcribe_audio(
            audio_path=temp_path,
            language=language,
            enable_diarization=enable_diarization,
            medical_context=medical_context
        )

        # Add request info
        result["request"] = {
            "filename": file.filename,
            "size_mb": round(file_size_mb, 2),
            "format": file_ext
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up: {temp_path}")
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("=" * 60)
    print("Whisper Transcription Server for HealthOS")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Language: {DEFAULT_LANGUAGE}")
    print(f"Diarization: {ENABLE_DIARIZATION}")
    print(f"Port: {PORT}")
    print("=" * 60)

    # Preload model
    try:
        load_whisper_model()
    except Exception as e:
        print(f"Warning: Could not preload model: {e}")
        print("Model will be loaded on first request")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True
    )
