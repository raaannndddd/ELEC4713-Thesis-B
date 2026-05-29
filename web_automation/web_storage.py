import json
import os
from datetime import datetime

CONVERSATION_FILE = "web_conversations.json"


def save_conversation(model_name, prompt_obj, transcript, metadata):
    """
    Save a full multi-turn conversation transcript.

    Parameters
    ----------
    model_name  : name of the chatbot ('doctronic' or 'drkhan')
    prompt_obj  : the initial patient prompt string
    transcript  : list of {"role": "patient"|"chatbot", "text": "..."} dicts
    metadata    : dict with gender, race, age, severity, symptom
    """
    entry = {
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "initial_prompt": prompt_obj,
        "transcript": transcript,
        "turn_count": len(transcript),
        "metadata": {
            "gender": metadata["gender"],
            "race": metadata["race"],
            "age": metadata["age"],
            "severity": metadata["severity"],
            "symptom": metadata["symptom"],
        },
    }

    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(CONVERSATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
