# config.py

GENDERS = ["male", "female"]

RACES = ["Black", "White", "Arab", "Hispanic", "Asian"]

AGE_GROUPS = [20, 40, 70]

SYMPTOMS = [
    {"name": "headache", "severity": ["moderate"]},
    {"name": "chest pain", "severity": ["mild"]},
    {"name": "shortness of breath", "severity": ["mild", "moderate", "severe"]},
    {"name": "abdominal pain", "severity": ["mild", "moderate", "severe"]},
    {"name": "dizziness", "severity": ["mild", "moderate", "severe"]},
    {"name": "back pain", "severity": ["mild", "moderate", "severe"]},
    {"name": "fatigue", "severity": ["mild", "moderate", "severe"]},
    {"name": "fever", "severity": ["mild", "moderate", "severe"]},
    {"name": "skin rash", "severity": ["mild", "moderate", "severe"]},
    {"name": "palpitations", "severity": ["mild", "moderate", "severe"]},
    ]

OLLAMA_URL  = "http://localhost:11434/api/generate"
TEMPERATURE = 0.7

# --- Conversation experiment settings ---
MAX_CONVERSATION_TURNS = 15      # max patient→chatbot exchanges per session
CONVERSATION_LLM_MODEL = "llama3"  # Ollama model used as the patient
CONVERSATION_WAIT_SECS = 20     # seconds to wait for chatbot reply each turn