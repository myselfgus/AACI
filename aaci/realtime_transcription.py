"""
Real-Time Audio Transcription with WebSocket Support
Streaming audio processing with Whisper, diarization, and ambient agent triggering.

Features:
- WebSocket endpoint for real-time audio streaming
- Voice Activity Detection (VAD)
- Streaming transcription with Faster-Whisper
- Real-time speaker diarization
- Ambient agent triggering during conversation
- Low-latency processing (<500ms)
"""

import asyncio
import logging
import json
import numpy as np
import soundfile as sf
import webrtcvad
from typing import Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
import tempfile

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Audio processing
import librosa
import noisereduce as nr

# Whisper
from faster_whisper import WhisperModel

# Diarization
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
except ImportError:
    DiarizationPipeline = None

# Ambient agents
from .ambient_agents import AmbientAgentManager, AgentType


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RealtimeConfig:
    """Configuration for real-time transcription."""
    # Model configuration
    whisper_model: str = "large-v3-turbo"
    compute_type: str = "float16"
    device: str = "cuda"
    language: str = "pt"

    # Audio configuration
    sample_rate: int = 16000
    chunk_duration_ms: int = 300  # 300ms chunks
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive

    # Processing configuration
    enable_noise_reduction: bool = True
    enable_diarization: bool = True
    enable_ambient_agents: bool = True
    min_speech_duration_ms: int = 500  # Minimum speech duration to process

    # Streaming configuration
    buffer_duration_s: int = 3  # Buffer audio for N seconds before processing
    overlap_duration_s: float = 0.5  # Overlap between chunks for context


# ============================================================================
# MODELS
# ============================================================================

class TranscriptionChunk(BaseModel):
    """Real-time transcription chunk."""
    text: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None
    is_final: bool = False
    timestamp: str = ""
    agents_triggered: List[Dict[str, str]] = []


class RealtimeStatus(BaseModel):
    """Status update for real-time connection."""
    status: str  # "connected", "processing", "error", "completed"
    message: str = ""
    timestamp: str = ""
    buffer_size: int = 0


# ============================================================================
# REAL-TIME TRANSCRIBER
# ============================================================================

class RealtimeTranscriber:
    """
    Real-time audio transcription with VAD, streaming Whisper, and diarization.

    Architecture:
    1. Receive audio chunks via WebSocket
    2. Apply Voice Activity Detection (VAD)
    3. Buffer audio until sufficient speech detected
    4. Transcribe with Faster-Whisper (streaming mode)
    5. Apply speaker diarization
    6. Trigger ambient agents based on content
    7. Send results back via WebSocket
    """

    def __init__(self, config: RealtimeConfig = None):
        """Initialize real-time transcriber."""
        self.config = config or RealtimeConfig()

        # Load Whisper model
        logger.info(f"Loading Whisper model: {self.config.whisper_model}")
        self.whisper_model = WhisperModel(
            self.config.whisper_model,
            device=self.config.device,
            compute_type=self.config.compute_type
        )

        # Initialize VAD
        self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)

        # Initialize diarization pipeline
        self.diarization_pipeline = None
        if self.config.enable_diarization and DiarizationPipeline:
            try:
                self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    "pyannote/speaker-diarization-3.1"
                )
                logger.info("Speaker diarization enabled")
            except Exception as e:
                logger.warning(f"Could not load diarization pipeline: {e}")

        # Initialize ambient agent manager
        self.ambient_manager = None
        if self.config.enable_ambient_agents:
            self.ambient_manager = AmbientAgentManager()
            logger.info("Ambient agent triggering enabled")

        # Buffers
        self.audio_buffer = []
        self.speech_buffer = []
        self.last_speech_time = 0
        self.total_audio_duration = 0

    async def process_websocket(self, websocket: WebSocket):
        """
        Main WebSocket handler for real-time transcription.

        Protocol:
        - Client sends: binary audio data (16-bit PCM, 16kHz)
        - Server sends: JSON with TranscriptionChunk or RealtimeStatus

        Args:
            websocket: FastAPI WebSocket connection
        """
        await websocket.accept()
        logger.info("WebSocket connection established")

        # Send initial status
        await self._send_status(
            websocket,
            "connected",
            "Real-time transcription started"
        )

        try:
            while True:
                # Receive audio chunk
                data = await websocket.receive_bytes()

                # Process audio chunk
                result = await self._process_audio_chunk(data)

                if result:
                    # Send transcription result
                    await websocket.send_json(result.dict())

                # Send periodic status updates
                if len(self.audio_buffer) % 10 == 0:
                    await self._send_status(
                        websocket,
                        "processing",
                        f"Buffer: {len(self.audio_buffer)} chunks",
                        buffer_size=len(self.audio_buffer)
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")

            # Process any remaining audio
            if self.speech_buffer:
                final_result = await self._transcribe_buffer()
                # Can't send after disconnect, but log for debugging
                logger.info(f"Final transcription: {final_result}")

        except Exception as e:
            logger.error(f"Error in WebSocket processing: {e}", exc_info=True)
            await self._send_status(
                websocket,
                "error",
                f"Processing error: {str(e)}"
            )
        finally:
            # Cleanup
            self._reset()

    async def _process_audio_chunk(self, audio_data: bytes) -> Optional[TranscriptionChunk]:
        """
        Process a single audio chunk.

        Args:
            audio_data: Raw audio bytes (16-bit PCM)

        Returns:
            TranscriptionChunk if speech detected and transcribed, None otherwise
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Add to buffer
        self.audio_buffer.append(audio_np)
        self.total_audio_duration += len(audio_np) / self.config.sample_rate

        # Voice Activity Detection
        is_speech = self._detect_speech(audio_data)

        if is_speech:
            self.speech_buffer.append(audio_np)
            self.last_speech_time = self.total_audio_duration

            # Check if buffer is full enough to transcribe
            if self._should_transcribe():
                return await self._transcribe_buffer()

        else:
            # Check if we should flush the buffer (silence after speech)
            silence_duration = self.total_audio_duration - self.last_speech_time
            if self.speech_buffer and silence_duration > 0.5:  # 500ms silence
                return await self._transcribe_buffer()

        return None

    def _detect_speech(self, audio_data: bytes) -> bool:
        """
        Detect speech in audio chunk using VAD.

        Args:
            audio_data: Raw audio bytes

        Returns:
            True if speech detected, False otherwise
        """
        # VAD requires specific chunk sizes (10, 20, or 30 ms)
        # We'll use 30ms chunks
        chunk_size = int(self.config.sample_rate * 0.03)  # 30ms

        try:
            # Check if we have enough data
            if len(audio_data) < chunk_size * 2:  # 16-bit = 2 bytes per sample
                return False

            # Take first 30ms for VAD check
            vad_chunk = audio_data[:chunk_size * 2]

            return self.vad.is_speech(vad_chunk, self.config.sample_rate)
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return True  # Assume speech on error to avoid missing content

    def _should_transcribe(self) -> bool:
        """Check if we have enough audio to transcribe."""
        if not self.speech_buffer:
            return False

        buffer_duration = sum(len(chunk) for chunk in self.speech_buffer) / self.config.sample_rate
        return buffer_duration >= self.config.buffer_duration_s

    async def _transcribe_buffer(self) -> Optional[TranscriptionChunk]:
        """
        Transcribe accumulated speech buffer.

        Returns:
            TranscriptionChunk with transcription result
        """
        if not self.speech_buffer:
            return None

        try:
            # Concatenate audio chunks
            audio = np.concatenate(self.speech_buffer)

            # Normalize audio
            audio = audio.astype(np.float32) / 32768.0

            # Apply noise reduction if enabled
            if self.config.enable_noise_reduction:
                audio = nr.reduce_noise(y=audio, sr=self.config.sample_rate)

            # Save to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio, self.config.sample_rate)
                temp_path = temp_file.name

            # Transcribe with Whisper
            segments, info = self.whisper_model.transcribe(
                temp_path,
                language=self.config.language,
                beam_size=5,
                vad_filter=True
            )

            # Collect segments
            transcription_text = ""
            start_time = self.last_speech_time - len(audio) / self.config.sample_rate
            end_time = self.last_speech_time
            avg_confidence = 0.0
            segment_count = 0

            for segment in segments:
                transcription_text += segment.text + " "
                avg_confidence += segment.avg_logprob
                segment_count += 1

            if segment_count > 0:
                avg_confidence /= segment_count
                # Convert log probability to confidence score
                confidence = min(1.0, max(0.0, np.exp(avg_confidence)))
            else:
                confidence = 0.0

            transcription_text = transcription_text.strip()

            # Speaker diarization (optional)
            speaker = None
            if self.diarization_pipeline:
                speaker = await self._identify_speaker(temp_path)

            # Trigger ambient agents
            agents_triggered = []
            if self.ambient_manager and transcription_text:
                triggered = self.ambient_manager.add_utterance(
                    transcription_text,
                    speaker=speaker or "unknown"
                )
                agents_triggered = [
                    {"agent": agent.value, "priority": params.get("priority", 0)}
                    for agent, params in triggered
                ]

            # Create result
            result = TranscriptionChunk(
                text=transcription_text,
                start=start_time,
                end=end_time,
                confidence=confidence,
                speaker=speaker,
                is_final=True,
                timestamp=datetime.now().isoformat(),
                agents_triggered=agents_triggered
            )

            # Clear speech buffer
            self.speech_buffer.clear()

            logger.info(f"Transcribed: {transcription_text[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None

    async def _identify_speaker(self, audio_path: str) -> Optional[str]:
        """
        Identify speaker using diarization pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            Speaker identifier (e.g., "SPEAKER_00")
        """
        if not self.diarization_pipeline:
            return None

        try:
            # Run diarization
            diarization = self.diarization_pipeline(audio_path)

            # Get the dominant speaker
            speaker_durations = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

            if speaker_durations:
                dominant_speaker = max(speaker_durations, key=speaker_durations.get)
                return dominant_speaker

        except Exception as e:
            logger.warning(f"Diarization error: {e}")

        return None

    async def _send_status(
        self,
        websocket: WebSocket,
        status: str,
        message: str,
        buffer_size: int = 0
    ):
        """Send status update via WebSocket."""
        status_obj = RealtimeStatus(
            status=status,
            message=message,
            timestamp=datetime.now().isoformat(),
            buffer_size=buffer_size
        )
        await websocket.send_json(status_obj.dict())

    def _reset(self):
        """Reset buffers and state."""
        self.audio_buffer.clear()
        self.speech_buffer.clear()
        self.last_speech_time = 0
        self.total_audio_duration = 0
        if self.ambient_manager:
            self.ambient_manager.reset()


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

# Global transcriber instance
_transcriber: Optional[RealtimeTranscriber] = None


def get_transcriber(config: RealtimeConfig = None) -> RealtimeTranscriber:
    """Get or create real-time transcriber instance."""
    global _transcriber
    if _transcriber is None:
        _transcriber = RealtimeTranscriber(config)
    return _transcriber


async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription.

    Usage:
        ws://localhost:8787/realtime

    Protocol:
        Client -> Server: Binary audio data (16-bit PCM, 16kHz)
        Server -> Client: JSON with transcription results

    Example client (JavaScript):
        const ws = new WebSocket('ws://localhost:8787/realtime');
        ws.binaryType = 'arraybuffer';

        // Send audio
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

        // Receive transcription
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Transcription:', data.text);
            if (data.agents_triggered && data.agents_triggered.length > 0) {
                console.log('Agents triggered:', data.agents_triggered);
            }
        };
    """
    transcriber = get_transcriber()
    await transcriber.process_websocket(websocket)
