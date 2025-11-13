"""
Medical and Psychiatric vocabulary specialized for Portuguese clinical contexts.
This module contains domain-specific terms and patterns for medical transcription.
"""

# Common medical terms in Portuguese (Expanded - November 2025)
MEDICAL_TERMS_PT = [
    # General medical terms
    "anamnese", "exame físico", "diagnóstico", "prognóstico", "tratamento",
    "sintomas", "sinais vitais", "pressão arterial", "frequência cardíaca",
    "temperatura", "saturação", "glicemia", "hemograma", "terapêutica",
    "evolução clínica", "conduta médica", "hipótese diagnóstica", "diagnóstico diferencial",
    "queixa principal", "história clínica", "antecedentes pessoais", "antecedentes familiares",
    "hábitos de vida", "alergias medicamentosas", "medicações em uso", "comorbidades",

    # Body systems
    "cardiovascular", "respiratório", "digestivo", "neurológico",
    "musculoesquelético", "endócrino", "renal", "hepático", "gastrointestinal",
    "genitourinário", "dermatológico", "oftalmológico", "otorrinolaringológico",
    "hematológico", "imunológico", "reumatológico", "oncológico",

    # Vital signs and measurements
    "pressão sistólica", "pressão diastólica", "pulso", "frequência respiratória",
    "temperatura axilar", "temperatura oral", "peso", "altura", "índice de massa corporal",
    "circunferência abdominal", "oximetria", "glicemia capilar", "glicemia de jejum",

    # Common conditions - Cardiovascular
    "hipertensão", "hipertensão arterial", "hipotensão", "cardiopatia", "valvopatia",
    "infarto", "infarto agudo do miocárdio", "angina estável", "angina instável",
    "acidente vascular cerebral", "AVC", "AVC isquêmico", "AVC hemorrágico",
    "insuficiência cardíaca", "insuficiência cardíaca congestiva",
    "arritmia", "fibrilação atrial", "flutter atrial", "taquicardia", "bradicardia",
    "endocardite", "miocardite", "pericardite", "trombose venosa profunda", "embolia pulmonar",
    "aterosclerose", "arteriosclerose", "doença arterial coronariana",

    # Respiratory conditions
    "asma", "bronquite", "bronquite crônica", "pneumonia", "pneumonia comunitária",
    "pneumonia hospitalar", "tuberculose", "derrame pleural", "pneumotórax",
    "doença pulmonar obstrutiva crônica", "DPOC", "enfisema", "fibrose pulmonar",
    "insuficiência respiratória", "dispneia", "tosse", "expectoração", "hemoptise",
    "rinite", "sinusite", "faringite", "laringite", "bronquiectasia",

    # Gastrointestinal conditions
    "gastrite", "úlcera péptica", "úlcera gástrica", "úlcera duodenal",
    "refluxo gastroesofágico", "doença do refluxo", "esofagite", "hérnia de hiato",
    "hepatite", "cirrose", "esteatose hepática", "insuficiência hepática",
    "pancreatite", "colecistite", "colelitíase", "cálculo biliar",
    "apendicite", "diverticulite", "doença inflamatória intestinal",
    "doença de Crohn", "retocolite ulcerativa", "síndrome do intestino irritável",
    "constipação", "diarreia", "náusea", "vômito", "hematêmese", "melena",
    "icterícia", "ascite", "hemorragia digestiva", "obstrução intestinal",

    # Endocrine and metabolic
    "diabetes", "diabetes mellitus", "diabetes tipo 1", "diabetes tipo 2",
    "hiperglicemia", "hipoglicemia", "cetoacidose diabética", "coma hiperosmolar",
    "hipotireoidismo", "hipertireoidismo", "tireoidite", "nódulo tireoidiano",
    "síndrome metabólica", "obesidade", "dislipidemia", "hipercolesterolemia",
    "hipertrigliceridemia", "gota", "hiperuricemia", "osteoporose", "osteopenia",

    # Renal and urological
    "insuficiência renal", "insuficiência renal aguda", "insuficiência renal crônica",
    "infecção do trato urinário", "ITU", "pielonefrite", "cistite", "uretrite",
    "litíase renal", "cálculo renal", "nefrolitíase", "hematúria", "proteinúria",
    "incontinência urinária", "retenção urinária", "hiperplasia prostática",

    # Neurological conditions
    "cefaleia", "enxaqueca", "migrânea", "cefaleia tensional", "cefaleia em salvas",
    "epilepsia", "convulsão", "crise epiléptica", "status epilepticus",
    "doença de Parkinson", "parkinsonismo", "tremor", "rigidez", "bradicinesia",
    "doença de Alzheimer", "demência", "demência vascular", "comprometimento cognitivo",
    "esclerose múltipla", "neuropatia", "neuropatia diabética", "neuropatia periférica",
    "paralisia", "paresia", "hemiplegia", "hemiparesia", "paraplegia", "tetraplegia",
    "tontura", "vertigem", "síncope", "perda de consciência", "coma",
    "meningite", "encefalite", "mielite", "neuralgia", "ciática",

    # Infectious diseases
    "infecção", "infecção bacteriana", "infecção viral", "infecção fúngica",
    "sepse", "choque séptico", "bacteremia", "viremia", "febre", "hipertermia",
    "dengue", "chikungunya", "zika", "COVID-19", "influenza", "gripe",
    "hepatite A", "hepatite B", "hepatite C", "HIV", "AIDS",
    "sífilis", "gonorreia", "clamídia", "herpes", "candidíase",

    # Hematological conditions
    "anemia", "anemia ferropriva", "anemia megaloblástica", "anemia falciforme",
    "leucemia", "linfoma", "mieloma múltiplo", "trombocitopenia", "trombocitose",
    "coagulopatia", "hemofilia", "púrpura", "epistaxe", "equimose", "hematoma",

    # Rheumatological and musculoskeletal
    "artrite", "artrite reumatoide", "osteoartrite", "osteoartrose", "artrose",
    "lúpus", "lúpus eritematoso sistêmico", "fibromialgia", "bursite", "tendinite",
    "lombalgia", "cervicalgia", "dorsalgia", "hérnia de disco", "ciática",
    "fratura", "entorse", "luxação", "contusão", "distensão muscular",

    # Dermatological conditions
    "dermatite", "eczema", "psoríase", "urticária", "prurido", "eritema",
    "lesão cutânea", "mácula", "pápula", "vesícula", "bolha", "pústula",
    "melanoma", "carcinoma basocelular", "carcinoma espinocelular",

    # Medications and drug classes
    "medicação", "medicamento", "posologia", "dosagem", "administração",
    "via oral", "via intramuscular", "via endovenosa", "via subcutânea", "via tópica",
    "anti-hipertensivo", "inibidor da ECA", "bloqueador dos canais de cálcio",
    "betabloqueador", "diurético", "tiazídico", "diurético de alça",
    "antibiótico", "penicilina", "cefalosporina", "quinolona", "macrolídeo",
    "analgésico", "anti-inflamatório", "anti-inflamatório não esteroidal", "AINE",
    "corticoide", "corticosteroide", "imunossupressor", "anti-histamínico",
    "anticoagulante", "antiagregante plaquetário", "estatina", "hipoglicemiante oral",
    "insulina", "broncodilatador", "inalador", "nebulização",
    "antiarrítmico", "vasodilatador", "antiemético", "antiácido", "inibidor de bomba de prótons",
    "laxante", "antidiarreico", "vitamina", "suplemento", "probiótico",

    # Procedures and exams
    "eletrocardiograma", "ECG", "raio-x", "radiografia", "tomografia computadorizada",
    "tomografia", "ressonância magnética", "ressonância", "ultrassonografia", "ultrassom",
    "ecocardiograma", "ecocardiografia", "endoscopia", "endoscopia digestiva alta",
    "colonoscopia", "retossigmoidoscopia", "biópsia", "punção", "paracentese",
    "toracocentese", "artrocentese", "espirometria", "teste ergométrico",
    "holter", "MAPA", "monitorização ambulatorial da pressão arterial",
    "exame de sangue", "hemograma completo", "leucograma", "plaquetas",
    "coagulograma", "tempo de protrombina", "INR", "TTPa",
    "bioquímica", "ureia", "creatinina", "sódio", "potássio", "cloro",
    "transaminases", "TGO", "TGP", "bilirrubinas", "fosfatase alcalina", "gama GT",
    "amilase", "lipase", "proteína C reativa", "PCR", "velocidade de hemossedimentação", "VHS",
    "hemoglobina glicada", "hemoglobina glicosilada", "curva glicêmica",
    "perfil lipídico", "colesterol total", "LDL", "HDL", "triglicerídeos",
    "hormônios tireoidianos", "TSH", "T3", "T4", "hormônio tireotrofina",
    "urocultura", "hemocultura", "cultura de escarro", "antibiograma",
    "sorologia", "teste rápido", "teste de antígeno", "PCR para vírus",
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

# Medical abbreviations commonly used in Portuguese clinical settings (Expanded)
MEDICAL_ABBREVIATIONS_PT = {
    # Vital signs
    "PA": "pressão arterial",
    "PAS": "pressão arterial sistólica",
    "PAD": "pressão arterial diastólica",
    "FC": "frequência cardíaca",
    "FR": "frequência respiratória",
    "Tax": "temperatura axilar",
    "Tor": "temperatura oral",
    "Tre": "temperatura retal",
    "SpO2": "saturação periférica de oxigênio",
    "SatO2": "saturação de oxigênio",
    "IMC": "índice de massa corporal",
    "BPM": "batimentos por minuto",
    "IRPM": "incursões respiratórias por minuto",

    # History and examination
    "HDA": "história da doença atual",
    "HPP": "história patológica pregressa",
    "HS": "história social",
    "HF": "história familiar",
    "EF": "exame físico",
    "ISDAS": "isento de sintomas, doença ou acidente específico",
    "QP": "queixa principal",
    "HD": "hipótese diagnóstica",
    "CD": "conduta",

    # Physical examination
    "AP": "ausculta pulmonar",
    "AC": "ausculta cardíaca",
    "AR": "ausculta respiratória",
    "BCRNF": "bulhas cardíacas rítmicas e normofonéticas",
    "MVF": "murmúrio vesicular fisiológico",
    "RHA": "ruídos hidroaéreos",
    "ABD": "abdome",
    "BEG": "bom estado geral",
    "REG": "regular estado geral",
    "MEG": "mau estado geral",
    "LOTE": "lúcido e orientado no tempo e espaço",
    "LOTEP": "lúcido, orientado no tempo, espaço e pessoa",

    # Cardiovascular
    "ACV": "acidente cerebrovascular",
    "AVE": "acidente vascular encefálico",
    "AVC": "acidente vascular cerebral",
    "IAM": "infarto agudo do miocárdio",
    "ICC": "insuficiência cardíaca congestiva",
    "IC": "insuficiência cardíaca",
    "FA": "fibrilação atrial",
    "DAC": "doença arterial coronariana",
    "TVP": "trombose venosa profunda",
    "TEP": "tromboembolismo pulmonar",
    "EP": "embolia pulmonar",

    # Respiratory
    "DPOC": "doença pulmonar obstrutiva crônica",
    "IRpA": "insuficiência respiratória aguda",
    "SARA": "síndrome da angústia respiratória aguda",
    "SDRA": "síndrome do desconforto respiratório agudo",
    "TB": "tuberculose",
    "BK": "bacilo de Koch",

    # Gastrointestinal
    "DRGE": "doença do refluxo gastroesofágico",
    "DII": "doença inflamatória intestinal",
    "DC": "doença de Crohn",
    "RCU": "retocolite ulcerativa",
    "HDA": "hemorragia digestiva alta",
    "HDB": "hemorragia digestiva baixa",

    # Endocrine and metabolic
    "DM": "diabetes mellitus",
    "DM1": "diabetes mellitus tipo 1",
    "DM2": "diabetes mellitus tipo 2",
    "HAS": "hipertensão arterial sistêmica",
    "SM": "síndrome metabólica",

    # Renal
    "ITU": "infecção do trato urinário",
    "IRA": "insuficiência renal aguda",
    "IRC": "insuficiência renal crônica",
    "DRC": "doença renal crônica",
    "NTA": "necrose tubular aguda",

    # Infectious diseases
    "DST": "doença sexualmente transmissível",
    "IST": "infecção sexualmente transmissível",
    "HIV": "vírus da imunodeficiência humana",
    "AIDS": "síndrome da imunodeficiência adquirida",
    "SIDA": "síndrome de imunodeficiência adquirida",

    # Laboratory
    "Hb": "hemoglobina",
    "Ht": "hematócrito",
    "Hto": "hematócrito",
    "Leuco": "leucócitos",
    "Plaq": "plaquetas",
    "VHS": "velocidade de hemossedimentação",
    "PCR": "proteína C reativa",
    "VR": "valores de referência",
    "Cr": "creatinina",
    "U": "ureia",
    "Na": "sódio",
    "K": "potássio",
    "Cl": "cloro",
    "HbA1c": "hemoglobina glicada",
    "INR": "razão normalizada internacional",
    "TP": "tempo de protrombina",
    "TTPa": "tempo de tromboplastina parcial ativada",
    "TGO": "transaminase glutâmico-oxalacética",
    "TGP": "transaminase glutâmico-pirúvica",
    "FA": "fosfatase alcalina",
    "GGT": "gama-glutamil transferase",
    "BD": "bilirrubina direta",
    "BI": "bilirrubina indireta",
    "BT": "bilirrubina total",

    # Imaging and procedures
    "RX": "raio-x",
    "TC": "tomografia computadorizada",
    "RM": "ressonância magnética",
    "USG": "ultrassonografia",
    "ECG": "eletrocardiograma",
    "ECO": "ecocardiograma",
    "EDA": "endoscopia digestiva alta",
    "COLO": "colonoscopia",
    "MAPA": "monitorização ambulatorial da pressão arterial",

    # Medications
    "VO": "via oral",
    "IV": "intravenoso",
    "IM": "intramuscular",
    "SC": "subcutâneo",
    "SL": "sublingual",
    "EV": "endovenoso",
    "ACO": "anticoncepcional oral",
    "AAS": "ácido acetilsalicílico",
    "AINE": "anti-inflamatório não esteroidal",
    "IECA": "inibidor da enzima conversora de angiotensina",
    "BRA": "bloqueador do receptor de angiotensina",
    "BCC": "bloqueador dos canais de cálcio",
    "BB": "betabloqueador",
    "IBP": "inibidor da bomba de prótons",

    # Time and frequency
    "QD": "uma vez ao dia",
    "BID": "duas vezes ao dia",
    "TID": "três vezes ao dia",
    "QID": "quatro vezes ao dia",
    "SOS": "se necessário",
    "PRN": "quando necessário",
    "AC": "antes das refeições",
    "PC": "pós-refeição",
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
