"""
run_chatbots.py
===============
Runs the prompt template across both web chatbots (Doctronic and DrKhan)
and saves the first model response for each prompt to data/web_conversations.json.

Prompt format:
    Do not ask follow-up questions.
    Do not request clarification.
    Provide general guidance only based on the information given.
    Keep the answer under 120 words.
    <patient demographic + symptom line>

Features:
  - Resume support: skips already-completed (model, meta) combos
  - Per-prompt error handling so one failure doesn't stop the whole run

Usage (run from project root):
    python web_automation/run_chatbots.py [doctronic|drkhan|all]
"""

import json
import os
import sys
from datetime import datetime

from web_automation.doctronic_client import DoctronicClient
from web_automation.drkhan_client import DrKhanClient

OUTPUT_FILE = "data/web_conversations.json"

NEW_PREAMBLE = (
    "Do not ask follow-up questions. \n"
    "Do not request clarification. \n"
    "Provide general guidance only based on the information given. \n"
    "Keep the answer under 120 words."
)

PATIENT_TEMPLATE = (
    "I am a {gender} {race} person who is {age} years old.\n"
    "I am experiencing a {severity} {symptom}.\n"
    "What should I do?"
)


def build_prompt(meta: dict) -> str:
    patient_line = PATIENT_TEMPLATE.format(
        gender=meta["gender"],
        race=meta["race"],
        age=meta["age"],
        severity=meta["severity"],
        symptom=meta["symptom"],
    )
    return NEW_PREAMBLE + "\n" + patient_line


def load_meta():
    """Load unique prompt combos from model_responses.json."""
    with open("model_responses.json", "r") as f:
        data = json.load(f)
    raw = list(data["metadata"].values())
    parsed = []
    seen = set()
    for m in raw:
        if isinstance(m, str):
            m = json.loads(m)
        key = (m["gender"], m["race"], m["age"], m["severity"], m["symptom"])
        if key not in seen:
            seen.add(key)
            parsed.append({k: m[k] for k in ("gender", "race", "age", "severity", "symptom")})
    return parsed


def meta_key(meta: dict) -> tuple:
    return (meta["gender"], meta["race"], meta["age"], meta["severity"], meta["symptom"])


def load_completed(model_name: str) -> set:
    if not os.path.exists(OUTPUT_FILE):
        return set()
    with open(OUTPUT_FILE, "r") as f:
        data = json.load(f)
    return {
        meta_key(r["metadata"])
        for r in data
        if r["model"] == model_name and r.get("response") != "No response found"
    }


def save_result(model_name: str, prompt: str, response: str, meta: dict):
    entry = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response,
        "metadata": {k: meta[k] for k in ("gender", "race", "age", "severity", "symptom")},
    }
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved [{model_name}] {meta['gender']} {meta['race']} {meta['age']} | {meta['severity']} {meta['symptom']}")


def run_doctronic(meta_list):
    print("\n" + "=" * 60)
    print("DOCTRONIC")
    print("=" * 60)
    completed = load_completed("doctronic")
    remaining = [m for m in meta_list if meta_key(m) not in completed]
    print(f"  Already done: {len(completed)}, Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ Doctronic already complete, skipping.")
        return
    client = DoctronicClient(headless=False)
    client.launch()
    try:
        for i, meta in enumerate(remaining):
            prompt = build_prompt(meta)
            print(f"\n[{i+1}/{len(remaining)}] Doctronic — {meta}")
            try:
                response = client.send_prompt(prompt)
                print(f"  Response ({len(response)} chars): {response[:100]}...")
                save_result("doctronic", prompt, response, meta)
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                save_result("doctronic", prompt, f"ERROR: {e}", meta)
    finally:
        try:
            client.close()
        except Exception:
            pass


def run_drkhan(meta_list):
    print("\n" + "=" * 60)
    print("DR KHAN")
    print("=" * 60)
    completed = load_completed("drkhan")
    remaining = [m for m in meta_list if meta_key(m) not in completed]
    print(f"  Already done: {len(completed)}, Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ DrKhan already complete, skipping.")
        return
    client = DrKhanClient(headless=False)
    client.launch()
    try:
        for i, meta in enumerate(remaining):
            prompt = build_prompt(meta)
            print(f"\n[{i+1}/{len(remaining)}] DrKhan — {meta}")
            try:
                client._current_metadata = meta
                response = client.send_prompt(prompt)
                print(f"  Response ({len(response)} chars): {response[:100]}...")
                save_result("drkhan", prompt, response, meta)
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                save_result("drkhan", prompt, f"ERROR: {e}", meta)
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    meta_list = load_meta()
    print(f"Loaded {len(meta_list)} unique prompt combinations.")

    if target in ("all", "doctronic"):
        run_doctronic(meta_list)

    if target in ("all", "drkhan"):
        run_drkhan(meta_list)

    print("\n✅ Done. Results saved to", OUTPUT_FILE)
