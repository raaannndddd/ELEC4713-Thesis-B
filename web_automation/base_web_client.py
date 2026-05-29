from playwright.sync_api import sync_playwright
import time


class BaseWebClient:
    def __init__(self, headless=False):
        self.headless = headless

    def launch(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()

    def close(self):
        self.browser.close()
        self.playwright.stop()

    # ------------------------------------------------------------------
    # Single-turn interface (original behaviour, kept for compatibility)
    # ------------------------------------------------------------------
    def send_prompt(self, prompt):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Multi-turn conversation interface
    # ------------------------------------------------------------------
    def open_chat(self):
        """Navigate to a fresh chat session. Subclasses should override."""
        raise NotImplementedError

    def send_message(self, text: str):
        """Type and submit a single message in the chat UI."""
        raise NotImplementedError

    def get_all_messages(self) -> list:
        """Return all visible chatbot response texts on the current page."""
        raise NotImplementedError

    def fill_form_if_present(self, metadata: dict):
        """
        Fill any in-chat demographic form that the chatbot may render
        before giving a medical response. Default: no-op.
        Subclasses override this when the chatbot requires form input.
        Returns True if a form was found and filled, False otherwise.
        """
        return False