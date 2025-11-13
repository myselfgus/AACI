"""
Ambient Agent Triggering System for Medical Consultations
Pattern matching and intelligent agent dispatching during real-time transcription.

This module enables ambient listening capabilities that detect specific patterns
in medical conversations and trigger appropriate agents or actions.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from fuzzywuzzy import fuzz, process


logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents that can be triggered during medical consultations."""
    # Clinical documentation agents
    SOAP_NOTE_GENERATOR = "soap_note_generator"
    PRESCRIPTION_WRITER = "prescription_writer"
    REFERRAL_CREATOR = "referral_creator"
    LAB_ORDER = "lab_order"

    # Decision support agents
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    DRUG_INTERACTION_CHECKER = "drug_interaction_checker"
    CLINICAL_GUIDELINE = "clinical_guideline"

    # Administrative agents
    APPOINTMENT_SCHEDULER = "appointment_scheduler"
    BILLING_CODER = "billing_coder"

    # Patient education agents
    PATIENT_EDUCATION = "patient_education"
    LIFESTYLE_COUNSELING = "lifestyle_counseling"

    # Emergency agents
    RED_FLAG_ALERT = "red_flag_alert"
    CRITICAL_LAB_ALERT = "critical_lab_alert"


class ConversationPhase(Enum):
    """Phases of a medical consultation."""
    GREETING = "greeting"
    CHIEF_COMPLAINT = "chief_complaint"
    HISTORY_PRESENT_ILLNESS = "history_present_illness"
    PAST_MEDICAL_HISTORY = "past_medical_history"
    MEDICATIONS = "medications"
    ALLERGIES = "allergies"
    SOCIAL_HISTORY = "social_history"
    FAMILY_HISTORY = "family_history"
    REVIEW_OF_SYSTEMS = "review_of_systems"
    PHYSICAL_EXAM = "physical_exam"
    ASSESSMENT = "assessment"
    PLAN = "plan"
    CLOSING = "closing"
    UNKNOWN = "unknown"


@dataclass
class PatternTrigger:
    """Defines a pattern that triggers an agent."""
    name: str
    pattern: str  # Regex pattern
    agent_type: AgentType
    priority: int = 1  # Higher = more urgent
    phase: Optional[ConversationPhase] = None
    fuzzy_match: bool = False
    fuzzy_threshold: int = 80
    context_window: int = 5  # Number of previous sentences to consider
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None


# Medical Pattern Triggers for Portuguese consultations
MEDICAL_PATTERNS = [
    # ===== RED FLAGS & EMERGENCIES =====
    PatternTrigger(
        name="chest_pain_emergency",
        pattern=r"(dor no peito|dor tor[áa]cica|aperto no peito|opressão tor[áa]cica)",
        agent_type=AgentType.RED_FLAG_ALERT,
        priority=10,
        parameters={"alert_type": "acute_coronary_syndrome"}
    ),
    PatternTrigger(
        name="stroke_symptoms",
        pattern=r"(perda de força|fraqueza (súbita|repentina)|dificuldade (de|para) falar|boca torta|paralisia facial)",
        agent_type=AgentType.RED_FLAG_ALERT,
        priority=10,
        parameters={"alert_type": "possible_stroke"}
    ),
    PatternTrigger(
        name="suicidal_ideation",
        pattern=r"(ideação suicida|pens(o|ei) em (me matar|suicídio)|vontade de morrer|não quero mais viver)",
        agent_type=AgentType.RED_FLAG_ALERT,
        priority=10,
        parameters={"alert_type": "suicide_risk"}
    ),
    PatternTrigger(
        name="severe_shortness_breath",
        pattern=r"(falta de ar (intensa|grave)|dispneia (aos|em) (mínimos esforços|repouso)|não consigo respirar)",
        agent_type=AgentType.RED_FLAG_ALERT,
        priority=9,
        parameters={"alert_type": "respiratory_distress"}
    ),

    # ===== PRESCRIPTION TRIGGERS =====
    PatternTrigger(
        name="medication_request",
        pattern=r"(vou prescrever|vamos iniciar|receitar|prescrição de|tomar|administrar).*(medicamento|remédio|comprimido)",
        agent_type=AgentType.PRESCRIPTION_WRITER,
        priority=7,
        phase=ConversationPhase.PLAN
    ),
    PatternTrigger(
        name="dose_adjustment",
        pattern=r"(aumentar|diminuir|ajustar|modificar).*(dose|dosagem|posologia)",
        agent_type=AgentType.PRESCRIPTION_WRITER,
        priority=7,
        phase=ConversationPhase.PLAN
    ),
    PatternTrigger(
        name="drug_allergy_mention",
        pattern=r"(alergia|al[ée]rgico|reação) (a|ao).*(medicamento|remédio|antibiótico|penicilina)",
        agent_type=AgentType.DRUG_INTERACTION_CHECKER,
        priority=8,
        phase=ConversationPhase.ALLERGIES
    ),

    # ===== LAB ORDER TRIGGERS =====
    PatternTrigger(
        name="lab_test_order",
        pattern=r"(solicitar|pedir|fazer|realizar).*(exame|hemograma|glicemia|colesterol|ureia|creatinina)",
        agent_type=AgentType.LAB_ORDER,
        priority=6,
        phase=ConversationPhase.PLAN
    ),
    PatternTrigger(
        name="imaging_order",
        pattern=r"(solicitar|pedir|fazer).*(raio-x|tomografia|ressonância|ultrassom|ecografia)",
        agent_type=AgentType.LAB_ORDER,
        priority=6,
        phase=ConversationPhase.PLAN
    ),

    # ===== REFERRAL TRIGGERS =====
    PatternTrigger(
        name="specialist_referral",
        pattern=r"(encaminhar|referenciar|encaminhamento) (para|ao).*(cardiologista|neurologista|psiquiatra|ortopedista|especialista)",
        agent_type=AgentType.REFERRAL_CREATOR,
        priority=6,
        phase=ConversationPhase.PLAN
    ),

    # ===== DIAGNOSIS SUPPORT =====
    PatternTrigger(
        name="diagnostic_uncertainty",
        pattern=r"(hipótese diagnóstica|diagnóstico diferencial|possibilidades|pode ser|suspeita de)",
        agent_type=AgentType.DIFFERENTIAL_DIAGNOSIS,
        priority=5,
        phase=ConversationPhase.ASSESSMENT
    ),
    PatternTrigger(
        name="multiple_medications",
        pattern=r"(toma|usa|faz uso de).*(vários|diversos|muitos).*(medicamentos|remédios)",
        agent_type=AgentType.DRUG_INTERACTION_CHECKER,
        priority=7,
        phase=ConversationPhase.MEDICATIONS
    ),

    # ===== PATIENT EDUCATION =====
    PatternTrigger(
        name="lifestyle_modification",
        pattern=r"(dieta|alimentação|exercício|atividade física|perder peso|parar de fumar|cessar tabagismo)",
        agent_type=AgentType.LIFESTYLE_COUNSELING,
        priority=4,
        phase=ConversationPhase.PLAN
    ),
    PatternTrigger(
        name="disease_explanation",
        pattern=r"(vou explicar|deixa eu explicar|sobre (sua|a) (doença|condição))",
        agent_type=AgentType.PATIENT_EDUCATION,
        priority=4,
        phase=ConversationPhase.ASSESSMENT
    ),

    # ===== APPOINTMENT SCHEDULING =====
    PatternTrigger(
        name="follow_up_scheduling",
        pattern=r"(retorno|retornar|voltar|próxima consulta) (em|daqui).*(dias|semanas|meses)",
        agent_type=AgentType.APPOINTMENT_SCHEDULER,
        priority=5,
        phase=ConversationPhase.CLOSING
    ),

    # ===== DOCUMENTATION TRIGGERS =====
    PatternTrigger(
        name="soap_note_completion",
        pattern=r"(concluir|finalizar|terminar).*(consulta|atendimento)",
        agent_type=AgentType.SOAP_NOTE_GENERATOR,
        priority=6,
        phase=ConversationPhase.CLOSING
    ),
]


# Fuzzy matching terms for medical concepts
FUZZY_MEDICAL_TERMS = {
    "hypertension": ["hipertensão", "pressão alta", "HAS", "hipertensão arterial"],
    "diabetes": ["diabetes", "diabetes mellitus", "DM", "açúcar no sangue alto"],
    "depression": ["depressão", "tristeza persistente", "humor deprimido", "TDM"],
    "anxiety": ["ansiedade", "TAG", "nervosismo", "preocupação excessiva"],
    "chest_pain": ["dor no peito", "dor torácica", "aperto no peito", "desconforto torácico"],
    "shortness_breath": ["falta de ar", "dispneia", "cansaço para respirar", "dificuldade respiratória"],
}


class AmbientAgentManager:
    """
    Manages ambient listening and agent triggering during medical consultations.

    Features:
    - Real-time pattern matching on transcribed text
    - Conversation phase detection
    - Context-aware agent triggering
    - Priority-based alert system
    - State machine for conversation flow
    """

    def __init__(self, patterns: List[PatternTrigger] = None):
        """
        Initialize the ambient agent manager.

        Args:
            patterns: List of pattern triggers (defaults to MEDICAL_PATTERNS)
        """
        self.patterns = patterns or MEDICAL_PATTERNS
        self.conversation_history: List[str] = []
        self.current_phase = ConversationPhase.UNKNOWN
        self.triggered_agents: List[Tuple[AgentType, Dict[str, Any]]] = []
        self.conversation_context: Dict[str, Any] = {}

    def add_utterance(self, text: str, speaker: str = "unknown") -> List[Tuple[AgentType, Dict[str, Any]]]:
        """
        Process a new utterance and trigger agents if patterns match.

        Args:
            text: Transcribed text
            speaker: Speaker identifier (e.g., "doctor", "patient")

        Returns:
            List of triggered agents with their parameters
        """
        self.conversation_history.append(text)

        # Update conversation phase
        self._update_phase(text)

        # Check for pattern matches
        triggered = []
        for pattern in self.patterns:
            if self._should_check_pattern(pattern):
                match = self._check_pattern(text, pattern)
                if match:
                    agent_info = (pattern.agent_type, {
                        **pattern.parameters,
                        "matched_text": match,
                        "speaker": speaker,
                        "phase": self.current_phase,
                        "priority": pattern.priority
                    })
                    triggered.append(agent_info)
                    self.triggered_agents.append(agent_info)
                    logger.info(f"Triggered agent: {pattern.agent_type.value} (Priority: {pattern.priority})")

                    # Execute callback if provided
                    if pattern.callback:
                        pattern.callback(match, self.conversation_context)

        return sorted(triggered, key=lambda x: x[1]["priority"], reverse=True)

    def _should_check_pattern(self, pattern: PatternTrigger) -> bool:
        """Check if pattern should be evaluated based on current phase."""
        if pattern.phase is None:
            return True
        return self.current_phase == pattern.phase

    def _check_pattern(self, text: str, pattern: PatternTrigger) -> Optional[str]:
        """
        Check if text matches the pattern.

        Args:
            text: Text to check
            pattern: Pattern trigger to match against

        Returns:
            Matched text if found, None otherwise
        """
        # Standard regex matching
        match = re.search(pattern.pattern, text.lower(), re.IGNORECASE)
        if match:
            return match.group(0)

        # Fuzzy matching if enabled
        if pattern.fuzzy_match:
            for key, terms in FUZZY_MEDICAL_TERMS.items():
                result = process.extractOne(text.lower(), terms, scorer=fuzz.partial_ratio)
                if result and result[1] >= pattern.fuzzy_threshold:
                    return result[0]

        return None

    def _update_phase(self, text: str):
        """Update the current conversation phase based on text content."""
        text_lower = text.lower()

        # Phase detection patterns
        phase_patterns = {
            ConversationPhase.CHIEF_COMPLAINT: r"(qual o motivo|o que traz|queixa principal|me conta o que)",
            ConversationPhase.HISTORY_PRESENT_ILLNESS: r"(quando começou|há quanto tempo|como começou|história da doença)",
            ConversationPhase.MEDICATIONS: r"(medicamentos|remédios|o que toma|usa algum medicamento)",
            ConversationPhase.ALLERGIES: r"(alergia|alérgico|tem alergia)",
            ConversationPhase.PHYSICAL_EXAM: r"(vou examinar|exame físico|deixa eu ver|auscultar)",
            ConversationPhase.ASSESSMENT: r"(diagnóstico|hipótese|conclusão|avaliação)",
            ConversationPhase.PLAN: r"(tratamento|plano|vamos fazer|conduta|prescrever)",
            ConversationPhase.CLOSING: r"(retorno|próxima consulta|qualquer coisa|dúvidas|pode ir)",
        }

        for phase, pattern in phase_patterns.items():
            if re.search(pattern, text_lower):
                self.current_phase = phase
                logger.debug(f"Conversation phase updated to: {phase.value}")
                break

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation and triggered agents."""
        return {
            "total_utterances": len(self.conversation_history),
            "current_phase": self.current_phase.value,
            "triggered_agents": [
                {"agent": agent.value, "params": params}
                for agent, params in self.triggered_agents
            ],
            "high_priority_alerts": [
                {"agent": agent.value, "params": params}
                for agent, params in self.triggered_agents
                if params.get("priority", 0) >= 8
            ]
        }

    def reset(self):
        """Reset the ambient agent manager for a new conversation."""
        self.conversation_history.clear()
        self.current_phase = ConversationPhase.UNKNOWN
        self.triggered_agents.clear()
        self.conversation_context.clear()


# Example custom callback function
def emergency_alert_callback(matched_text: str, context: Dict[str, Any]):
    """Callback for emergency situations."""
    logger.critical(f"EMERGENCY ALERT: {matched_text}")
    # Here you would integrate with alert systems, notifications, etc.
    context["emergency_detected"] = True
    context["emergency_text"] = matched_text


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize manager
    manager = AmbientAgentManager()

    # Simulate a conversation
    conversation = [
        ("doctor", "Bom dia! Qual o motivo da consulta hoje?"),
        ("patient", "Doutor, estou com dor no peito há 2 horas."),
        ("doctor", "A dor é forte? Irradia para o braço?"),
        ("patient", "Sim, está irradiando para o braço esquerdo e estou suando muito."),
        ("doctor", "Vou solicitar um eletrocardiograma urgente e exames de troponina."),
        ("doctor", "Vamos encaminhar você para o cardiologista imediatamente."),
    ]

    for speaker, text in conversation:
        print(f"\n{speaker.upper()}: {text}")
        agents = manager.add_utterance(text, speaker)
        if agents:
            print(f"🤖 Agents triggered: {[a[0].value for a in agents]}")

    # Get summary
    summary = manager.get_conversation_summary()
    print(f"\n📊 Conversation Summary:")
    print(f"Total utterances: {summary['total_utterances']}")
    print(f"Current phase: {summary['current_phase']}")
    print(f"High priority alerts: {len(summary['high_priority_alerts'])}")
