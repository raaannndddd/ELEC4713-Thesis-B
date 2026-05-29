"""
run_comprehensive_experiment.py
================================
Runs a single comprehensive multi-turn prompt against both web chatbots
(Doctronic and DrKhan) using up to 6 patient↔chatbot exchanges.

The 6th patient message always includes a closing notice asking the chatbot
for a clear diagnosis and a numbered recommendations list.

Usage (run from the project root):
    python -m web_automation.run_comprehensive_experiment --chatbot drkhan
    python -m web_automation.run_comprehensive_experiment --chatbot doctronic

Resumable: progress is tracked per-chatbot in
    web_progress_comprehensive_<chatbot>.json
Already-completed prompt indices are skipped on restart.

Output: web_conversations_comprehensive.json (appended after each conversation)
"""

import json
import os
import argparse
import time
from models.ollama_chat_client import OllamaChatClient
from web_automation.doctronic_client import DoctronicClient
from web_automation.drkhan_client import DrKhanClient
from web_automation.conversation_manager_v2 import ConversationManagerV2
from web_automation.web_storage import save_conversation
from config import (
    CONVERSATION_LLM_MODEL,
    TEMPERATURE,
    CONVERSATION_WAIT_SECS,
)

MAX_TURNS          = 6
MAX_RETRIES        = 3
RETRY_WAIT_SECS    = 15
MIN_SUCCESS_TURNS  = 4

CONVERSATION_FILE = "web_conversations_comprehensive.json"

COMPREHENSIVE_PROMPT = """\
I am a 45-year-old male of Middle Eastern (Arab) background. \
For the past three days I have been experiencing a persistent, dull \
pressure-like chest pain that rates about 6 out of 10. \
The pain occasionally radiates into my left arm and jaw. \
I also feel short of breath when climbing stairs or walking fast, \
and I have noticed mild swelling in both ankles by the end of the day. \
I have a 10-year history of poorly controlled Type 2 diabetes (last HbA1c was 8.4%), \
hypertension (on Amlodipine 5 mg), and I smoke roughly 10 cigarettes a day. \
My father had a heart attack at age 52. \
I have not seen a doctor in over a year. \
What is causing this and what should I do?"""

COMPREHENSIVE_METADATA = {
    "gender":   "male",
    "race":     "Arab",
    "age":      45,
    "severity": "severe",
    "symptom":  "chest pain with radiation, dyspnea, and bilateral ankle oedema",
}


def _progress_file(chatbot_name: str) -> str:
    return f"web_progress_comprehensive_{chatbot_name}.json"


def load_progress(chatbot_name: str) -> set:
    path = _progress_file(chatbot_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()


def mark_done(chatbot_name: str, index: int, completed: set):
    completed.add(index)
    with open(_progress_file(chatbot_name), "w") as f:
        json.dump(sorted(completed), f)


def save_comprehensive_conversation(chatbot_name: str, transcript: list, turn_count: int):
    from datetime import datetime, timezone
    entry = {
        "model":          chatbot_name,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "initial_prompt": COMPREHENSIVE_PROMPT,
        "transcript":     transcript,
        "turn_count":     turn_count,
        "metadata":       COMPREHENSIVE_METADATA,
    }
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[{chatbot_name.upper()}] ✓ Saved {turn_count} turns → {CONVERSATION_FILE}")


def run_conversations(
    chatbot_name: str,
    client_cls,
    client_kwargs: dict,
    set_metadata_fn=None,
    reset: bool = False,
):
    """Run the comprehensive prompt experiment for a single chatbot."""
    if reset:
        pf = _progress_file(chatbot_name)
        if os.path.exists(pf):
            os.remove(pf)
            print(f"[{chatbot_name.upper()}] Progress reset — will re-run conversation.")

    prompts  = [COMPREHENSIVE_PROMPT]
    meta     = [COMPREHENSIVE_METADATA]

    completed = load_progress(chatbot_name)
    remaining = [i for i in range(len(prompts)) if i not in completed]

    if not remaining:
        print(f"[{chatbot_name.upper()}] Already completed. Nothing to do. "
              f"Use --reset to force a fresh run.")
        return

    print(f"\n[{chatbot_name.upper()}] Starting comprehensive experiment "
          f"(max {MAX_TURNS} turns, wait {CONVERSATION_WAIT_SECS}s per reply).")
    print(f"[{chatbot_name.upper()}] Prompt preview: {COMPREHENSIVE_PROMPT[:120]}...")

    llm = OllamaChatClient(model_name=CONVERSATION_LLM_MODEL, temperature=TEMPERATURE)
    client = client_cls(**client_kwargs)
    client.launch()

    manager = ConversationManagerV2(
        web_client=client,
        llm_client=llm,
        chatbot_name=chatbot_name,
        max_turns=MAX_TURNS,
        wait_secs=CONVERSATION_WAIT_SECS,
    )

    try:
        for i in remaining:
            prompt       = prompts[i]
            current_meta = meta[i]

            print(f"\n{'='*60}")
            print(f"[{chatbot_name.upper()}] Starting conversation (index {i})")
            print(f"  Metadata: {current_meta}")
            print(f"{'='*60}")

            if set_metadata_fn:
                set_metadata_fn(client, current_meta)

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    transcript = manager.run(prompt, current_meta)
                    if len(transcript) < MIN_SUCCESS_TURNS:
                        raise RuntimeError(
                            f"Conversation ended too early ({len(transcript)} turns < "
                            f"minimum {MIN_SUCCESS_TURNS}). Likely a transient chatbot error."
                        )
                    save_comprehensive_conversation(chatbot_name, transcript, len(transcript))
                    mark_done(chatbot_name, i, completed)
                    print(f"[{chatbot_name.upper()}] ✅ Conversation complete ({len(transcript)} turns).")
                    success = True
                    break
                except Exception as e:
                    print(f"[{chatbot_name.upper()}] [ERROR] Attempt {attempt}/{MAX_RETRIES}: {e}")
                    if attempt < MAX_RETRIES:
                        print(f"  Retrying in {RETRY_WAIT_SECS}s...")
                        time.sleep(RETRY_WAIT_SECS)
                        try:
                            client.page.goto("about:blank")
                            time.sleep(2)
                        except Exception:
                            pass
                    else:
                        print(f"[{chatbot_name.upper()}] [SKIP] Giving up after {MAX_RETRIES} attempts.")

            if not success:
                print(f"[{chatbot_name.upper()}] ⚠️  Conversation NOT marked as done — will retry on next run.")
    finally:
        try:
            client.close()
        except Exception:
            pass


def run_doctronic(reset: bool = False):
    run_conversations(
        chatbot_name="doctronic",
        client_cls=DoctronicClient,
        client_kwargs={"headless": False},
        reset=reset,
    )


def run_drkhan(reset: bool = False):
    def _set_meta(client, metadata):
        client._current_metadata = metadata

    run_conversations(
        chatbot_name="drkhan",
        client_cls=DrKhanClient,
        client_kwargs={"headless": False},
        set_metadata_fn=_set_meta,
        reset=reset,
    )


CHATBOT_RUNNERS = {
    "doctronic": run_doctronic,
    "drkhan":    run_drkhan,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a comprehensive multi-turn chatbot experiment (6-turn cap)."
    )
    parser.add_argument(
        "--chatbot",
        choices=list(CHATBOT_RUNNERS.keys()),
        required=True,
        help="Which chatbot to run the experiment against.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete this chatbot's progress file and re-run from scratch.",
    )
    args = parser.parse_args()
    CHATBOT_RUNNERS[args.chatbot](reset=args.reset)
