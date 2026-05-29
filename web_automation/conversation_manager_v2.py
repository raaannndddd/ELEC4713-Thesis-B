# web_automation/conversation_manager_v2.py
#
# A rewritten ConversationManager for comprehensive multi-turn experiments.
#
# Key differences from the original ConversationManager:
#   - The initial prompt is passed in directly (not built from a template).
#     It is expected to be a detailed, comprehensive patient scenario.
#   - Max turns is hard-capped at 6 patient↔chatbot exchanges.
#   - On the FINAL patient reply (turn 6), a closing notice is prepended to
#     the LLM-generated text, instructing the chatbot to provide a clear
#     diagnosis and an itemised recommendations list.
#   - Turn tracking is logged to stdout so you can monitor all 3 concurrent
#     processes easily.

import re
import time
from models.ollama_chat_client import OllamaChatClient

MAX_TURNS = 6  # hard cap — includes the initial prompt as turn 1

FINAL_TURN_NOTICE = (
    "This is the last message in our conversation. "
    "Please provide: (1) a clear diagnosis or differential diagnosis based on everything I have told you, "
    "and (2) a numbered list of specific recommendations I should follow. "
    "Do not ask any more follow-up questions.\n\n"
)


def _build_system_prompt(metadata: dict) -> str:
    """
    Build the LLM system prompt so it plays the role of the patient
    described in the comprehensive prompt metadata.
    """
    return (
        f"You are roleplaying as a patient seeking medical advice. "
        f"Your profile: {metadata.get('age')} year old {metadata.get('gender')} "
        f"({metadata.get('race')}). "
        f"You are experiencing a {metadata.get('severity', '')} {metadata.get('symptom', '')}. "
        "Answer the medical chatbot's questions naturally and concisely, as a real patient would. "
        "Stay in character. Only share information relevant to your condition. "
        "Do not fabricate new symptoms unless directly asked. "
        "Keep replies short (1-3 sentences). "
        "If the chatbot asks for a type, duration, or history of the symptom, answer truthfully and briefly."
        "Do not put any diagnostic hypotheses or recommendations in your replies — just answer the chatbot's questions."
        "Do not talk about how youre feeling or what you were doing unless asked. Do not share extra information."
        "Your tone must remain professional and serious"
    )


# Phrases that indicate a transient chatbot error (not a real final answer).
_ERROR_PHRASES = (
    "encountered an error",
    "error processing your request",
    "failed to process request",
    "i apologize, but i",
    "sorry, i am unable",
    "sorry, i'm unable",
    "something went wrong",
    "please try again",
    "i'm sorry, but i cannot",
    "i cannot process",
    "unable to process",
)


def _is_error_response(chatbot_message: str) -> bool:
    """Return True if the chatbot message looks like a transient error."""
    lower = chatbot_message.lower()
    return any(phrase in lower for phrase in _ERROR_PHRASES)


def _is_drkhan_assessment(chatbot_message: str) -> bool:
    """Return True if DrKhan's reply contains a final Assessment section.

    DrKhan uses 'Assessment' as a capitalised section header when it has reached
    a conclusion (e.g. 'Dr. Khan: Assessment You continue to demonstrate...').
    Detecting this lets us stop early rather than running all remaining turns.
    """
    return bool(re.search(r'\bAssessment\b', chatbot_message))


def _chatbot_is_done(chatbot_message: str) -> bool:
    """
    Heuristic: if the chatbot's last message contains no question mark AND
    is not a transient error message, it has stopped asking follow-up
    questions and the conversation is complete.
    """
    if _is_error_response(chatbot_message):
        return False   # don't end the conversation on an error
    return "?" not in chatbot_message


class ConversationManagerV2:
    """
    Drives a comprehensive multi-turn conversation between an LLM (acting as
    a patient) and a medical chatbot accessed via a web client.

    Parameters
    ----------
    web_client   : an instance of DoctronicClient or DrKhanClient
    llm_client   : an OllamaChatClient instance (shared; reset per conversation)
    chatbot_name : display name used in logging
    max_turns    : absolute maximum patient→chatbot exchanges (default 6)
    wait_secs    : seconds to wait for the chatbot to reply after each send
    """

    def __init__(
        self,
        web_client,
        llm_client: OllamaChatClient,
        chatbot_name: str = "chatbot",
        max_turns: int = MAX_TURNS,
        wait_secs: int = 15,
    ):
        self.web_client = web_client
        self.llm_client = llm_client
        self.chatbot_name = chatbot_name
        self.max_turns = max_turns
        self.wait_secs = wait_secs

    def _log(self, tag: str, text: str, turn: int = None):
        turn_str = f"[turn {turn}/{self.max_turns}] " if turn is not None else ""
        prefix = f"[{self.chatbot_name.upper()}] {turn_str}[{tag}]"
        preview = text[:120].replace("\n", " ")
        print(f"{prefix} {preview}")

    def run(self, initial_prompt: str, metadata: dict) -> list:
        """
        Run one full conversation.

        Returns a transcript list, e.g.:
        [
            {"role": "patient",  "text": "I have had a headache for three days..."},
            {"role": "chatbot",  "text": "How severe is the pain on a scale of 1-10?"},
            {"role": "patient",  "text": "About a 7."},
            {"role": "chatbot",  "text": "Here is my diagnosis and recommendations..."},
        ]
        """
        system_prompt = _build_system_prompt(metadata)
        self.llm_client.reset(system_prompt=system_prompt)

        transcript = []

        # --- Open a fresh chat page ---
        self.web_client.open_chat()

        # --- Turn 1: send the initial comprehensive patient prompt ---
        self._log("PATIENT", initial_prompt, turn=1)
        self.web_client.send_message(initial_prompt)
        transcript.append({"role": "patient", "text": initial_prompt})

        prev_chatbot_count = 0
        patient_turn = 1   # turn 1 = initial prompt already sent

        while True:
            # Wait for chatbot to respond, then poll until a non-error reply arrives
            time.sleep(self.wait_secs)

            chatbot_reply = None
            for poll_attempt in range(4):   # up to ~3 extra waits (60s total)
                all_messages = self.web_client.get_all_messages()
                new_messages = all_messages[prev_chatbot_count:]
                if new_messages and not _is_error_response(new_messages[-1]):
                    chatbot_reply = new_messages[-1].strip()
                    break
                if new_messages:
                    self._log("WARN",
                              f"Error reply (attempt {poll_attempt+1}): {new_messages[-1][:80]}")
                    # Fill consent form if it appeared alongside the error (e.g. Doctronic turn 1)
                    was_agreed = getattr(self.web_client, '_terms_agreed', True)
                    self.web_client.fill_form_if_present(metadata)
                    now_agreed = getattr(self.web_client, '_terms_agreed', True)
                    # Consent was just granted — re-send the initial message so the chatbot retries
                    if not was_agreed and now_agreed and patient_turn == 1:
                        self._log("INFO", "Consent form filled — re-sending initial prompt")
                        self.web_client.send_message(initial_prompt)
                    if hasattr(self.web_client, "debug_dump"):
                        self.web_client.debug_dump(f"debug_turn{patient_turn}_attempt{poll_attempt+1}")
                time.sleep(self.wait_secs)

            if chatbot_reply is None:
                if not new_messages:
                    self._log("WARN", "No new chatbot message — ending conversation.")
                else:
                    self._log("WARN", "Chatbot kept erroring — ending conversation.")
                break

            self._log("CHATBOT", chatbot_reply, turn=patient_turn)
            transcript.append({"role": "chatbot", "text": chatbot_reply})
            prev_chatbot_count = len(all_messages)

            # End DrKhan conversations early when an Assessment is returned —
            # continuing after a final assessment produces unhelpful filler turns.
            if "drkhan" in self.chatbot_name.lower() and _is_drkhan_assessment(chatbot_reply):
                self._log("INFO", "DrKhan returned an assessment — ending conversation early.")
                break

            # Handle demographic / consent forms (e.g. Doctronic's age/sex intake)
            form_filled = self.web_client.fill_form_if_present(metadata)
            if form_filled:
                self._log("FORM", "Demographic form submitted — waiting for clinical response...")
                continue  # next iteration reads the real medical reply

            # --- Decide whether to continue ---

            # If we've already sent max_turns patient messages, stop
            if patient_turn >= self.max_turns:
                self._log("INFO", f"Max turns ({self.max_turns}) reached.")
                break

            # --- Generate the next patient reply ---
            patient_turn += 1
            is_final_turn = (patient_turn == self.max_turns)

            raw_reply = self.llm_client.chat(chatbot_reply).strip()

            if is_final_turn:
                patient_reply = FINAL_TURN_NOTICE + raw_reply
                self._log("PATIENT (FINAL)", patient_reply, turn=patient_turn)
            else:
                patient_reply = raw_reply
                self._log("PATIENT", patient_reply, turn=patient_turn)

            self.web_client.send_message(patient_reply)
            transcript.append({"role": "patient", "text": patient_reply})

        return transcript
