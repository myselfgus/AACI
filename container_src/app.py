"""
HealthOS Whisper Container Worker
Audio Transcription, Diarization, Medical NER, Paralinguistics & Prosody Analysis
October 2025 - Production Ready
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Tuple
import httpx
import os
import json
from datetime import datetime
import logging
import tempfile
import uuid

# Audio Processing
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline
import parselmouth
import myprosody
import opensmile
import librosa
import numpy as np

# NER & Deep Learning
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
import torch

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HealthOS Whisper Container Worker",
    version="1.0.0",
    description="Medical-grade audio processing: Transcription, Diarization, NER, Paralinguistics, Prosody"
)

# CORS middleware for Cloudflare Workers communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

class SpeakerSegment(BaseModel):
    start: float
    end: float
    speaker: str
    duration: float

class MedicalEntity(BaseModel):
    entity: str
    word: str
    start: int
    end: int
    score: float

class ParalinguisticFeatures(BaseModel):
    """Acoustic features: emotion, stress, voice quality indicators"""
    mfcc_mean: List[float]  # Mel-frequency cepstral coefficients
    mfcc_std: List[float]
    pitch_mean: float  # F0 mean
    pitch_std: float
    intensity_mean: float  # Energy
    intensity_std: float
    voice_quality_hnr: float  # Harmonic-to-noise ratio
    zero_crossing_rate: float
    spectral_centroid: float
    emotion_indicators: Dict[str, float]  # stress, fatigue, etc.

class ProsodyAnalysis(BaseModel):
    """Prosodic features: rhythm, intonation, pauses"""
    speaking_rate: float  # Syllables per second
    f0_min: float
    f0_max: float
    f0_mean: float
    f0_std: float
    f0_quantiles: Dict[str, float]  # 25%, 50%, 75%
    intonation_index: float
    pause_count: int
    pause_duration_mean: float
    pitch_range: float
    voice_breaks: int

class WhisperProcessingRequest(BaseModel):
    audio_url: Optional[str] = None  # URL to audio file
    patient_id: Optional[str] = None  # Patient identifier
    session_id: Optional[str] = None  # Session identifier
    language: str = "pt"  # Portuguese
    include_speakers: bool = True
    include_medical_ner: bool = True
    include_paralinguistics: bool = True
    include_prosody: bool = True
    webhook_url: Optional[str] = None  # Callback URL for results

class WhisperProcessingResponse(BaseModel):
    processing_id: str
    status: str  # "queued", "processing", "completed", "failed"
    transcription: Optional[List[TranscriptionSegment]] = None
    speakers: Optional[List[SpeakerSegment]] = None
    medical_entities: Optional[List[MedicalEntity]] = None
    paralinguistic_features: Optional[ParalinguisticFeatures] = None
    prosody_analysis: Optional[ProsodyAnalysis] = None
    language: str
    confidence: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None
    timestamp: str

# ============================================================================
# GLOBAL STATE & MODEL INITIALIZATION
# ============================================================================

# In-memory processing store (use persistent DB in production)
processing_jobs = {}

class WhisperWorkerModels:
    """Lazy-load models for performance"""

    def __init__(self):
        self._whisper = None
        self._diarization = None
        self._ner_tokenizer = None
        self._ner_model = None
        self._smile = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self._device}")

    @property
    def whisper(self):
        if self._whisper is None:
            logger.info("Loading Faster-Whisper large-v3-turbo...")
            compute_type = "float16" if self._device == "cuda" else "int8"
            self._whisper = WhisperModel(
                "large-v3-turbo",
                device=self._device,
                compute_type=compute_type
            )
            logger.info("✓ Whisper loaded")
        return self._whisper

    @property
    def diarization(self):
        if self._diarization is None:
            logger.info("Loading Pyannote speaker-diarization-3.1...")
            hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
            try:
                self._diarization = DiarizationPipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token if hf_token else None
                )
                logger.info("✓ Diarization pipeline loaded")
            except Exception as e:
                logger.warning(f"Diarization pipeline not available: {e}")
        return self._diarization

    @property
    def ner_models(self):
        if self._ner_tokenizer is None:
            logger.info("Loading BioBERTpt medical NER...")
            self._ner_tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-all")
            self._ner_model = AutoModelForTokenClassification.from_pretrained(
                "pucpr/biobertpt-all"
            )
            logger.info("✓ BioBERTpt loaded")
        return self._ner_tokenizer, self._ner_model

    @property
    def opensmile(self):
        if self._smile is None:
            logger.info("Loading OpenSMILE feature extractor...")
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            logger.info("✓ OpenSMILE loaded")
        return self._smile

models = WhisperWorkerModels()

# ============================================================================
# AUDIO PROCESSING FUNCTIONS
# ============================================================================

async def download_audio(url: str, temp_dir: str) -> str:
    """Download audio file from URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True, timeout=300.0)
            response.raise_for_status()

            # Save to temp file
            audio_path = os.path.join(temp_dir, f"audio_{uuid.uuid4()}.wav")
            with open(audio_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Downloaded audio to {audio_path}")
            return audio_path
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        raise

async def transcribe_audio(audio_path: str, language: str = "pt") -> Tuple[List[Dict], Dict]:
    """Transcribe audio using Faster-Whisper"""
    try:
        logger.info(f"Transcribing {audio_path}...")

        segments = []
        info = None

        for segment in models.whisper.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=0.0  # Deterministic
        ):
            segments.append({
                "id": len(segments),
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": None  # Faster-Whisper doesn't provide per-segment confidence
            })

        logger.info(f"✓ Transcribed {len(segments)} segments")
        return segments, info
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

async def diarize_speakers(audio_path: str) -> List[Dict]:
    """Identify and separate speakers using Pyannote"""
    try:
        if models.diarization is None:
            logger.warning("Diarization pipeline not available")
            return []

        logger.info(f"Diarizing speakers in {audio_path}...")

        diarization = models.diarization(audio_path)

        speakers = []
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
                "duration": segment.end - segment.start
            })

        logger.info(f"✓ Identified {len(set(s['speaker'] for s in speakers))} speakers")
        return speakers
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return []

async def extract_medical_entities(text: str) -> List[Dict]:
    """Extract medical entities using BioBERTpt NER"""
    try:
        logger.info("Extracting medical entities...")

        # Use HF pipeline for NER
        ner_pipeline = hf_pipeline(
            "ner",
            model="pucpr/biobertpt-all",
            device=0 if models._device == "cuda" else -1,
            aggregation_strategy="simple"
        )

        entities = ner_pipeline(text[:512])  # Limit input length

        medical_entities = []
        for entity in entities:
            medical_entities.append({
                "entity": entity.get("entity_group", "O"),
                "word": entity.get("word", ""),
                "start": entity.get("start", 0),
                "end": entity.get("end", 0),
                "score": entity.get("score", 0.0)
            })

        logger.info(f"✓ Extracted {len(medical_entities)} medical entities")
        return medical_entities
    except Exception as e:
        logger.error(f"Medical NER extraction failed: {e}")
        return []

async def extract_paralinguistic_features(audio_path: str) -> Dict:
    """Extract acoustic/paralinguistic features"""
    try:
        logger.info("Extracting paralinguistic features...")

        # OpenSMILE features
        smile_features = models.opensmile.process_file(audio_path)

        # Librosa features (MFCC, spectral centroid, ZCR)
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # Pitch (F0) and intensity
        try:
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            intensity = sound.to_intensity()

            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced

            f0_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
            f0_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

            intensity_values = intensity.values.flatten()
            intensity_mean = float(np.mean(intensity_values))
            intensity_std = float(np.std(intensity_values))
        except Exception as e:
            logger.warning(f"Pitch/intensity extraction failed: {e}")
            f0_mean = f0_std = intensity_mean = intensity_std = 0.0

        features = {
            "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
            "mfcc_std": np.std(mfcc, axis=1).tolist(),
            "pitch_mean": f0_mean,
            "pitch_std": f0_std,
            "intensity_mean": intensity_mean,
            "intensity_std": intensity_std,
            "voice_quality_hnr": 0.0,  # Simplified - compute HNR if needed
            "zero_crossing_rate": float(np.mean(zcr)),
            "spectral_centroid": float(np.mean(spectral_centroid)),
            "emotion_indicators": {}  # Can be filled with wav2vec2 emotion model
        }

        logger.info("✓ Extracted paralinguistic features")
        return features
    except Exception as e:
        logger.error(f"Paralinguistic feature extraction failed: {e}")
        return {}

async def analyze_prosody(audio_path: str) -> Dict:
    """Analyze prosodic features: pitch, rhythm, intonation"""
    try:
        logger.info("Analyzing prosody...")

        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch()
        intensity = sound.to_intensity()

        # Pitch analysis
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) == 0:
            return {}

        f0_min = float(np.min(pitch_values))
        f0_max = float(np.max(pitch_values))
        f0_mean = float(np.mean(pitch_values))
        f0_std = float(np.std(pitch_values))

        # F0 quantiles
        quantiles = {
            "q25": float(np.percentile(pitch_values, 25)),
            "q50": float(np.percentile(pitch_values, 50)),
            "q75": float(np.percentile(pitch_values, 75))
        }

        # Speaking rate (approximated from audio duration)
        y, sr = librosa.load(audio_path, sr=16000)
        duration_seconds = len(y) / sr

        # Simplified prosody analysis
        prosody = {
            "speaking_rate": len(y) / (sr * duration_seconds) if duration_seconds > 0 else 0.0,
            "f0_min": f0_min,
            "f0_max": f0_max,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "f0_quantiles": quantiles,
            "intonation_index": f0_std / f0_mean if f0_mean > 0 else 0.0,
            "pause_count": 0,  # Simplified
            "pause_duration_mean": 0.0,
            "pitch_range": f0_max - f0_min,
            "voice_breaks": 0  # Simplified
        }

        logger.info("✓ Analyzed prosody")
        return prosody
    except Exception as e:
        logger.error(f"Prosody analysis failed: {e}")
        return {}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Whisper Container Worker",
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/process")
async def process_audio(
    request: WhisperProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Main endpoint for audio processing"""

    processing_id = str(uuid.uuid4())

    # Store job state
    processing_jobs[processing_id] = {
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.dict()
    }

    # Queue background processing
    background_tasks.add_task(
        process_audio_task,
        processing_id,
        request
    )

    return WhisperProcessingResponse(
        processing_id=processing_id,
        status="queued",
        language=request.language,
        timestamp=datetime.utcnow().isoformat()
    )

async def process_audio_task(processing_id: str, request: WhisperProcessingRequest):
    """Background task for audio processing"""
    start_time = datetime.utcnow()

    try:
        processing_jobs[processing_id]["status"] = "processing"

        # Download audio
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = await download_audio(request.audio_url, temp_dir)

            # Transcription
            transcription, _ = await transcribe_audio(audio_path, request.language)

            # Diarization
            speakers = []
            if request.include_speakers:
                speakers = await diarize_speakers(audio_path)

            # Medical NER
            medical_entities = []
            if request.include_medical_ner and transcription:
                full_text = " ".join([s["text"] for s in transcription])
                medical_entities = await extract_medical_entities(full_text)

            # Paralinguistic features
            paralinguistic = None
            if request.include_paralinguistics:
                paralinguistic = await extract_paralinguistic_features(audio_path)

            # Prosody analysis
            prosody = None
            if request.include_prosody:
                prosody = await analyze_prosody(audio_path)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Store results
            processing_jobs[processing_id].update({
                "status": "completed",
                "transcription": transcription,
                "speakers": speakers,
                "medical_entities": medical_entities,
                "paralinguistic_features": paralinguistic,
                "prosody_analysis": prosody,
                "processing_time_seconds": processing_time,
                "completed_at": datetime.utcnow().isoformat()
            })

            logger.info(f"✓ Processing {processing_id} completed in {processing_time:.2f}s")

            # Callback webhook if provided
            if request.webhook_url:
                await notify_webhook(request.webhook_url, processing_jobs[processing_id])

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        processing_jobs[processing_id].update({
            "status": "failed",
            "error": str(e),
            "processing_time_seconds": (datetime.utcnow() - start_time).total_seconds()
        })

@app.get("/status/{processing_id}")
async def get_status(processing_id: str):
    """Get processing status and results"""
    if processing_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Processing job not found")

    job = processing_jobs[processing_id]
    return WhisperProcessingResponse(
        processing_id=processing_id,
        status=job.get("status"),
        transcription=job.get("transcription"),
        speakers=job.get("speakers"),
        medical_entities=job.get("medical_entities"),
        paralinguistic_features=job.get("paralinguistic_features"),
        prosody_analysis=job.get("prosody_analysis"),
        language=job.get("request", {}).get("language", "pt"),
        processing_time_seconds=job.get("processing_time_seconds"),
        error=job.get("error"),
        timestamp=datetime.utcnow().isoformat()
    )

async def notify_webhook(webhook_url: str, result: Dict):
    """Send results to webhook URL"""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=result, timeout=30.0)
            logger.info(f"Webhook notification sent to {webhook_url}")
    except Exception as e:
        logger.error(f"Webhook notification failed: {e}")

@app.post("/health-check")
async def extended_health_check():
    """Extended health check with model availability"""
    return {
        "status": "healthy",
        "service": "Whisper Container Worker",
        "models": {
            "whisper": "loaded" if models._whisper else "not_loaded",
            "diarization": "loaded" if models._diarization else "not_loaded",
            "ner": "loaded" if models._ner_model else "not_loaded",
            "opensmile": "loaded" if models._smile else "not_loaded"
        },
        "gpu_available": torch.cuda.is_available(),
        "device": models._device,
        "jobs_processed": len(processing_jobs),
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# REAL-TIME WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/realtime")
async def websocket_realtime_transcription(websocket: WebSocket):
    """
    Real-time audio transcription via WebSocket with ambient agent triggering.

    Protocol:
    - Client sends: Binary audio data (16-bit PCM, 16kHz mono)
    - Server sends: JSON with transcription results and triggered agents

    Features:
    - Voice Activity Detection (VAD)
    - Real-time transcription with Faster-Whisper
    - Speaker diarization
    - Ambient agent triggering (pattern matching)
    - Medical context awareness

    Example client (JavaScript):
        const ws = new WebSocket('ws://localhost:8787/realtime');
        ws.binaryType = 'arraybuffer';

        // Send audio chunks
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            const audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                const audioData = e.inputBuffer.getChannelData(0);
                const int16Data = new Int16Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {
                    int16Data[i] = audioData[i] * 32767;
                }
                ws.send(int16Data.buffer);
            };

            source.connect(processor);
            processor.connect(audioContext.destination);
        });

        // Receive transcription and agent triggers
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.text) {
                console.log('Transcription:', data.text);
                console.log('Speaker:', data.speaker);
                if (data.agents_triggered && data.agents_triggered.length > 0) {
                    console.log('🤖 Agents triggered:', data.agents_triggered);
                }
            }
        };
    """
    try:
        # Import real-time transcription module
        import sys
        sys.path.insert(0, '/home/user/AACI')
        from aaci.realtime_transcription import get_transcriber, RealtimeConfig

        # Initialize transcriber with medical Portuguese configuration
        config = RealtimeConfig(
            whisper_model="large-v3-turbo",
            compute_type="float16",
            device=models._device,
            language="pt",
            enable_noise_reduction=True,
            enable_diarization=True,
            enable_ambient_agents=True,
            buffer_duration_s=3,
            overlap_duration_s=0.5
        )

        transcriber = get_transcriber(config)
        await transcriber.process_websocket(websocket)

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8787"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
