"""
Medical and Psychiatric vocabulary specialized for Portuguese clinical contexts.
This module contains domain-specific terms and patterns for medical transcription.
"""

# Common medical terms in Portuguese
MEDICAL_TERMS_PT = [
    # General medical terms
    "anamnese", "exame físico", "diagnóstico", "prognóstico", "tratamento",
    "sintomas", "sinais vitais", "pressão arterial", "frequência cardíaca",
    "temperatura", "saturação", "glicemia", "hemograma",
    
    # Body systems
    "cardiovascular", "respiratório", "digestivo", "neurológico", 
    "musculoesquelético", "endócrino", "renal", "hepático",
    
    # Common conditions
    "hipertensão", "diabetes", "asma", "bronquite", "pneumonia",
    "infarto", "acidente vascular cerebral", "AVC", "insuficiência cardíaca",
    "arritmia", "fibrilação atrial", "angina",
    
    # Medications
    "medicação", "medicamento", "posologia", "dosagem", "administração",
    "via oral", "via intramuscular", "via endovenosa", "via subcutânea",
    "anti-hipertensivo", "antibiótico", "analgésico", "anti-inflamatório",
    
    # Procedures
    "eletrocardiograma", "ECG", "raio-x", "tomografia", "ressonância magnética",
    "ultrassonografia", "endoscopia", "colonoscopia", "biópsia",
]

# Psychiatric terms in Portuguese
PSYCHIATRIC_TERMS_PT = [
    # General psychiatric terms
    "psiquiatria", "psicoterapia", "saúde mental", "transtorno mental",
    "avaliação psiquiátrica", "história psiquiátrica",
    
    # Mood disorders
    "depressão", "transtorno depressivo", "depressão maior", "distimia",
    "transtorno bipolar", "mania", "hipomania", "episódio maníaco",
    "ciclotimia", "humor", "afeto", "eutimia",
    
    # Anxiety disorders
    "ansiedade", "transtorno de ansiedade", "transtorno de pânico",
    "agorafobia", "fobia social", "fobia específica", "transtorno de ansiedade generalizada",
    "TAG", "ataque de pânico",
    
    # Psychotic disorders
    "esquizofrenia", "psicose", "delírio", "alucinação", "pensamento desorganizado",
    "sintomas positivos", "sintomas negativos", "transtorno esquizoafetivo",
    
    # Personality and behavior
    "transtorno de personalidade", "borderline", "personalidade antissocial",
    "narcisismo", "comportamento", "insight", "juízo crítico",
    
    # Treatment and assessment
    "psicofármaco", "antidepressivo", "antipsicótico", "ansiolítico",
    "estabilizador de humor", "benzodiazepínico", "ISRS", "inibidor seletivo",
    "terapia cognitivo-comportamental", "TCC", "psicanálise",
    "exame do estado mental", "orientação", "memória", "atenção", "concentração",
    
    # Symptoms and signs
    "insônia", "hipersonia", "anedonia", "apatia", "ideação suicida",
    "tentativa de suicídio", "automutilação", "impulsividade", "agitação",
    "retardo psicomotor", "pensamento acelerado", "fuga de ideias",
]

# Medical abbreviations commonly used in Portuguese clinical settings
MEDICAL_ABBREVIATIONS_PT = {
    "PA": "pressão arterial",
    "FC": "frequência cardíaca",
    "FR": "frequência respiratória",
    "Tax": "temperatura axilar",
    "SpO2": "saturação periférica de oxigênio",
    "IMC": "índice de massa corporal",
    "HDA": "história da doença atual",
    "HPP": "história patológica pregressa",
    "HS": "história social",
    "HF": "história familiar",
    "EF": "exame físico",
    "AP": "ausculta pulmonar",
    "AC": "ausculta cardíaca",
    "ACV": "acidente cerebrovascular",
    "IAM": "infarto agudo do miocárdio",
    "ICC": "insuficiência cardíaca congestiva",
    "DM": "diabetes mellitus",
    "HAS": "hipertensão arterial sistêmica",
    "DPOC": "doença pulmonar obstrutiva crônica",
    "ITU": "infecção do trato urinário",
    "AVE": "acidente vascular encefálico",
}

# Psychiatric abbreviations
PSYCHIATRIC_ABBREVIATIONS_PT = {
    "TDM": "transtorno depressivo maior",
    "TB": "transtorno bipolar",
    "TAG": "transtorno de ansiedade generalizada",
    "TOC": "transtorno obsessivo-compulsivo",
    "TEPT": "transtorno de estresse pós-traumático",
    "TDAH": "transtorno de déficit de atenção e hiperatividade",
    "TEA": "transtorno do espectro autista",
    "TBP": "transtorno de personalidade borderline",
    "TPA": "transtorno de personalidade antissocial",
    "TCC": "terapia cognitivo-comportamental",
    "ISRS": "inibidor seletivo da recaptação de serotonina",
    "ADT": "antidepressivo tricíclico",
    "IMAO": "inibidor da monoamina oxidase",
}

def get_all_medical_terms():
    """Get all medical terms combined."""
    return MEDICAL_TERMS_PT + PSYCHIATRIC_TERMS_PT

def get_all_abbreviations():
    """Get all medical abbreviations combined."""
    return {**MEDICAL_ABBREVIATIONS_PT, **PSYCHIATRIC_ABBREVIATIONS_PT}

def expand_abbreviations(text: str) -> str:
    """
    Expand medical abbreviations in text.
    
    Args:
        text: Input text with potential abbreviations
        
    Returns:
        Text with expanded abbreviations
    """
    all_abbrevs = get_all_abbreviations()
    result = text
    for abbrev, expansion in all_abbrevs.items():
        result = result.replace(f" {abbrev} ", f" {expansion} ")
        result = result.replace(f" {abbrev}.", f" {expansion}.")
        result = result.replace(f" {abbrev},", f" {expansion},")
    return result
