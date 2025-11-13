"""
API Communication Schemas for AACI
Data models and interfaces for agent integration and external systems.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class TranscriptionStatus(str, Enum):
    """Status of transcription processing."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SpeakerRole(str, Enum):
    """Speaker role in medical consultation."""
    DOCTOR = "doctor"
    PATIENT = "patient"
    NURSE = "nurse"
    FAMILY_MEMBER = "family_member"
    UNKNOWN = "unknown"


class MedicalSpecialty(str, Enum):
    """Medical specialties."""
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    PSYCHIATRY = "psychiatry"
    GENERAL_PRACTICE = "general_practice"
    EMERGENCY = "emergency"
    ONCOLOGY = "oncology"
    PEDIATRICS = "pediatrics"
    OTHER = "other"


class AgentPriority(str, Enum):
    """Priority levels for agent actions."""
    CRITICAL = "critical"  # 10: Emergency situations
    HIGH = "high"  # 7-9: Important clinical actions
    MEDIUM = "medium"  # 4-6: Standard documentation/orders
    LOW = "low"  # 1-3: Educational/administrative


# ============================================================================
# REQUEST MODELS
# ============================================================================

class TranscriptionRequest(BaseModel):
    """Request for audio transcription."""
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    language: str = Field("pt", description="Language code (ISO 639-1)")
    enable_diarization: bool = Field(True, description="Enable speaker diarization")
    enable_medical_ner: bool = Field(True, description="Enable medical NER")
    enable_paralinguistics: bool = Field(False, description="Enable emotion/stress analysis")
    enable_ambient_agents: bool = Field(True, description="Enable ambient agent triggering")
    medical_specialty: Optional[MedicalSpecialty] = Field(None, description="Medical specialty context")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for async results")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('language')
    def validate_language(cls, v):
        supported = ['pt', 'en', 'es']
        if v not in supported:
            raise ValueError(f"Language must be one of {supported}")
        return v


class RealtimeSessionConfig(BaseModel):
    """Configuration for real-time WebSocket session."""
    language: str = Field("pt", description="Language code")
    sample_rate: int = Field(16000, description="Audio sample rate (Hz)")
    enable_diarization: bool = Field(True, description="Enable speaker diarization")
    enable_ambient_agents: bool = Field(True, description="Enable ambient agents")
    vad_aggressiveness: int = Field(2, description="VAD aggressiveness (0-3)")
    buffer_duration_s: float = Field(3.0, description="Audio buffer duration (seconds)")
    medical_specialty: Optional[MedicalSpecialty] = Field(None, description="Medical specialty")

    @validator('vad_aggressiveness')
    def validate_vad(cls, v):
        if not 0 <= v <= 3:
            raise ValueError("VAD aggressiveness must be between 0 and 3")
        return v


class AgentActionRequest(BaseModel):
    """Request to execute an agent action."""
    agent_type: str = Field(..., description="Type of agent to trigger")
    context: Dict[str, Any] = Field(..., description="Context data for agent")
    priority: AgentPriority = Field(AgentPriority.MEDIUM, description="Action priority")
    consultation_id: Optional[str] = Field(None, description="Consultation identifier")
    user_id: Optional[str] = Field(None, description="User identifier")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class TranscriptionSegment(BaseModel):
    """Single transcription segment."""
    id: int = Field(..., description="Segment ID")
    start: float = Field(..., description="Start time (seconds)")
    end: float = Field(..., description="End time (seconds)")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score (0-1)")
    speaker: Optional[str] = Field(None, description="Speaker identifier")
    speaker_role: Optional[SpeakerRole] = Field(None, description="Speaker role")
    medical_terms: Optional[List[str]] = Field(default_factory=list, description="Detected medical terms")


class SpeakerSegment(BaseModel):
    """Speaker diarization segment."""
    start: float = Field(..., description="Start time (seconds)")
    end: float = Field(..., description="End time (seconds)")
    speaker: str = Field(..., description="Speaker identifier (e.g., SPEAKER_00)")
    speaker_role: Optional[SpeakerRole] = Field(None, description="Detected speaker role")
    duration: float = Field(..., description="Duration (seconds)")
    turn_index: Optional[int] = Field(None, description="Turn index in conversation")


class MedicalEntity(BaseModel):
    """Medical named entity."""
    entity_type: str = Field(..., description="Entity type (SYMPTOM, MEDICATION, etc.)")
    text: str = Field(..., description="Entity text")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    confidence: float = Field(..., description="Confidence score (0-1)")
    normalized_term: Optional[str] = Field(None, description="Normalized medical term")
    snomed_code: Optional[str] = Field(None, description="SNOMED CT code")


class ParalinguisticFeatures(BaseModel):
    """Paralinguistic and acoustic features."""
    pitch_mean: float = Field(..., description="Mean pitch (F0) in Hz")
    pitch_std: float = Field(..., description="Pitch standard deviation")
    pitch_range: float = Field(..., description="Pitch range (max-min)")
    intensity_mean: float = Field(..., description="Mean intensity (dB)")
    speaking_rate: float = Field(..., description="Speaking rate (syllables/second)")
    pause_count: int = Field(..., description="Number of pauses")
    pause_duration_mean: float = Field(..., description="Mean pause duration (seconds)")
    voice_quality_hnr: float = Field(..., description="Harmonic-to-Noise Ratio")
    emotion_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Emotion/stress indicators (stress, anxiety, etc.)"
    )


class AgentTrigger(BaseModel):
    """Triggered agent information."""
    agent_type: str = Field(..., description="Type of agent triggered")
    agent_name: str = Field(..., description="Human-readable agent name")
    priority: int = Field(..., description="Priority level (1-10)")
    priority_label: AgentPriority = Field(..., description="Priority label")
    matched_text: str = Field(..., description="Text that triggered the agent")
    matched_pattern: Optional[str] = Field(None, description="Pattern that matched")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Trigger timestamp")
    recommended_action: Optional[str] = Field(None, description="Recommended action description")


class TranscriptionResponse(BaseModel):
    """Complete transcription response."""
    transcription_id: str = Field(..., description="Unique transcription ID")
    status: TranscriptionStatus = Field(..., description="Processing status")
    segments: List[TranscriptionSegment] = Field(default_factory=list, description="Transcription segments")
    speakers: List[SpeakerSegment] = Field(default_factory=list, description="Speaker diarization")
    medical_entities: List[MedicalEntity] = Field(default_factory=list, description="Medical entities")
    paralinguistic_features: Optional[ParalinguisticFeatures] = Field(None, description="Paralinguistic analysis")
    agents_triggered: List[AgentTrigger] = Field(default_factory=list, description="Triggered agents")
    full_text: str = Field("", description="Complete transcription text")
    language: str = Field("pt", description="Detected/configured language")
    duration: float = Field(..., description="Audio duration (seconds)")
    processing_time: float = Field(..., description="Processing time (seconds)")
    word_count: int = Field(0, description="Total word count")
    medical_term_count: int = Field(0, description="Medical terms detected")
    conversation_phase: Optional[str] = Field(None, description="Current conversation phase")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class RealtimeTranscriptionChunk(BaseModel):
    """Real-time transcription chunk (WebSocket)."""
    text: str = Field(..., description="Transcribed text chunk")
    start: float = Field(..., description="Start time (seconds)")
    end: float = Field(..., description="End time (seconds)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    speaker: Optional[str] = Field(None, description="Speaker identifier")
    is_final: bool = Field(False, description="Is this chunk final?")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    agents_triggered: List[AgentTrigger] = Field(default_factory=list, description="Agents triggered by this chunk")


class SessionStatus(BaseModel):
    """Real-time session status update."""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status")
    message: str = Field("", description="Status message")
    buffer_size: int = Field(0, description="Current buffer size")
    total_chunks_processed: int = Field(0, description="Total chunks processed")
    total_duration: float = Field(0.0, description="Total audio duration processed (seconds)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Status timestamp")


class AgentActionResponse(BaseModel):
    """Response from agent action execution."""
    action_id: str = Field(..., description="Action identifier")
    agent_type: str = Field(..., description="Agent type")
    status: str = Field(..., description="Execution status (success, failed, pending)")
    result: Dict[str, Any] = Field(default_factory=dict, description="Action result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")


# ============================================================================
# AGENT-SPECIFIC SCHEMAS
# ============================================================================

class SOAPNote(BaseModel):
    """SOAP (Subjective, Objective, Assessment, Plan) clinical note."""
    subjective: str = Field("", description="Subjective findings (patient's description)")
    objective: str = Field("", description="Objective findings (exam, vitals)")
    assessment: str = Field("", description="Assessment/diagnosis")
    plan: str = Field("", description="Treatment plan")
    consultation_date: datetime = Field(default_factory=datetime.now)
    doctor_name: Optional[str] = None
    patient_id: Optional[str] = None
    icd10_codes: List[str] = Field(default_factory=list, description="ICD-10 diagnosis codes")
    generated_from_audio: bool = Field(True, description="Generated automatically from audio")


class Prescription(BaseModel):
    """Medical prescription."""
    medications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of medications with name, dosage, frequency"
    )
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    consultation_date: datetime = Field(default_factory=datetime.now)
    instructions: str = Field("", description="Special instructions")
    duration_days: Optional[int] = Field(None, description="Treatment duration")
    pharmacy_instructions: Optional[str] = Field(None, description="Instructions for pharmacist")


class LabOrder(BaseModel):
    """Laboratory or imaging order."""
    order_type: str = Field(..., description="Type: lab, imaging, procedure")
    tests: List[str] = Field(..., description="List of tests/exams ordered")
    urgency: str = Field("routine", description="Urgency: stat, urgent, routine")
    clinical_indication: str = Field("", description="Clinical indication for tests")
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    order_date: datetime = Field(default_factory=datetime.now)


class ReferralRequest(BaseModel):
    """Referral to specialist."""
    specialty: str = Field(..., description="Specialist type")
    reason: str = Field(..., description="Reason for referral")
    urgency: str = Field("routine", description="Urgency level")
    clinical_summary: str = Field("", description="Brief clinical summary")
    patient_id: Optional[str] = None
    referring_doctor_id: Optional[str] = None
    referral_date: datetime = Field(default_factory=datetime.now)


class ClinicalAlert(BaseModel):
    """Clinical alert/red flag."""
    alert_type: str = Field(..., description="Type of alert")
    severity: AgentPriority = Field(..., description="Alert severity")
    description: str = Field(..., description="Alert description")
    matched_criteria: List[str] = Field(..., description="Criteria that triggered alert")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    timestamp: datetime = Field(default_factory=datetime.now)
    requires_immediate_attention: bool = Field(False, description="Requires immediate attention")


# ============================================================================
# WEBHOOK SCHEMAS
# ============================================================================

class WebhookPayload(BaseModel):
    """Webhook payload for async transcription completion."""
    event_type: str = Field(..., description="Event type (transcription.completed, agent.triggered, etc.)")
    transcription_id: str = Field(..., description="Transcription ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    data: Union[TranscriptionResponse, AgentTrigger, ClinicalAlert] = Field(
        ...,
        description="Event data"
    )


# ============================================================================
# BATCH PROCESSING SCHEMAS
# ============================================================================

class BatchTranscriptionRequest(BaseModel):
    """Batch transcription request for multiple files."""
    audio_files: List[str] = Field(..., description="List of audio file URLs or paths")
    shared_config: TranscriptionRequest = Field(..., description="Shared configuration")
    batch_id: Optional[str] = Field(None, description="Batch identifier")


class BatchTranscriptionStatus(BaseModel):
    """Batch transcription status."""
    batch_id: str = Field(..., description="Batch identifier")
    total_files: int = Field(..., description="Total files in batch")
    completed: int = Field(0, description="Completed transcriptions")
    failed: int = Field(0, description="Failed transcriptions")
    in_progress: int = Field(0, description="In progress")
    results: List[TranscriptionResponse] = Field(default_factory=list, description="Completed results")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Error details")


# ============================================================================
# HEALTH & MONITORING SCHEMAS
# ============================================================================

class HealthCheck(BaseModel):
    """System health check response."""
    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    service_name: str = Field("AACI Whisper Service")
    version: str = Field("1.0.0")
    timestamp: datetime = Field(default_factory=datetime.now)
    models: Dict[str, str] = Field(
        default_factory=dict,
        description="Model statuses (loaded, not_loaded, error)"
    )
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_memory_used: Optional[float] = Field(None, description="GPU memory used (GB)")
    gpu_memory_total: Optional[float] = Field(None, description="Total GPU memory (GB)")
    active_sessions: int = Field(0, description="Active real-time sessions")
    jobs_queued: int = Field(0, description="Queued jobs")
    jobs_processing: int = Field(0, description="Jobs in progress")


class Metrics(BaseModel):
    """Service metrics."""
    total_transcriptions: int = Field(0, description="Total transcriptions processed")
    total_audio_hours: float = Field(0.0, description="Total audio hours processed")
    average_processing_time: float = Field(0.0, description="Average processing time (seconds)")
    average_wer: Optional[float] = Field(None, description="Average Word Error Rate")
    agents_triggered_total: int = Field(0, description="Total agents triggered")
    critical_alerts: int = Field(0, description="Critical alerts raised")
    uptime_seconds: float = Field(0.0, description="Service uptime (seconds)")
    last_updated: datetime = Field(default_factory=datetime.now)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example request
    request = TranscriptionRequest(
        audio_url="https://example.com/consult.mp3",
        language="pt",
        enable_diarization=True,
        enable_medical_ner=True,
        enable_ambient_agents=True,
        medical_specialty=MedicalSpecialty.CARDIOLOGY
    )

    print("Example Request:")
    print(request.json(indent=2))

    # Example response
    response = TranscriptionResponse(
        transcription_id="550e8400-e29b-41d4-a716-446655440000",
        status=TranscriptionStatus.COMPLETED,
        segments=[
            TranscriptionSegment(
                id=0,
                start=0.0,
                end=5.2,
                text="Bom dia, doutor. Estou com dor no peito há dois dias.",
                confidence=0.95,
                speaker="SPEAKER_01",
                speaker_role=SpeakerRole.PATIENT,
                medical_terms=["dor no peito"]
            )
        ],
        agents_triggered=[
            AgentTrigger(
                agent_type="red_flag_alert",
                agent_name="Red Flag Alert - Chest Pain",
                priority=10,
                priority_label=AgentPriority.CRITICAL,
                matched_text="dor no peito",
                recommended_action="Immediate ECG and cardiac evaluation"
            )
        ],
        duration=300.0,
        processing_time=8.3
    )

    print("\nExample Response:")
    print(response.json(indent=2))
