import time
from web_automation.base_web_client import BaseWebClient


class DrKhanClient(BaseWebClient):

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def open_chat(self):
        """
        Full onboarding flow for drkhan.ai:
          1. Land on homepage → click "Start Free Consultation"
          2. Fill age + select gender from the Radix combobox
          3. Click "Start Consultation" → wait for chat page
        Metadata (age, gender) must be stored on self._current_metadata
        before this method is called.
        """
        metadata = getattr(self, "_current_metadata", {})
        age     = str(metadata.get("age", "30"))
        gender_raw = metadata.get("gender", "male")
        # Radix combobox options are capitalised ("Male" / "Female")
        gender  = gender_raw.capitalize()

        # Close and relaunch the browser to get a completely fresh session.
        # We keep the Playwright instance alive (can't restart sync_playwright
        # mid-process) and only restart the browser itself.
        self.browser.close()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        self.page.goto("https://drkhan.ai")
        time.sleep(4)

        # 1. Click "Start Free Consultation"
        # There are two matching buttons (with and without the emoji).
        # Prefer the emoji one, fall back to the plain-text variant.
        start_btn = self.page.get_by_role("button", name="🩺 Start Free Consultation")
        if start_btn.count() == 0:
            start_btn = self.page.get_by_role("button", name="Start Free Consultation", exact=True)
        start_btn.first.click()
        time.sleep(3)

        # 2a. Fill age — wait for the field to appear before filling
        age_input = self.page.locator("input[placeholder='Enter your age (18+)']")
        age_input.wait_for(state="visible", timeout=60_000)
        age_input.fill(age)
        time.sleep(0.3)

        # 2b. Open the gender combobox and pick the right option
        self.page.locator("button[role='combobox']").click()
        time.sleep(0.5)
        # Radix renders options as role="option" inside a listbox
        self.page.get_by_role("option", name=gender, exact=True).click()
        time.sleep(0.3)

        # 3. Click "Start Consultation"
        self.page.locator("button:has-text('Start Consultation')").click()
        time.sleep(4)   # wait for chat UI to load

    # ------------------------------------------------------------------
    # Multi-turn conversation interface
    # ------------------------------------------------------------------
    def send_message(self, text: str):
        """Type and submit a message in the Dr Khan chat textarea."""
        textarea = self.page.locator("textarea[placeholder='Type your message...']")
        textarea.fill(text)
        send_btn = self.page.locator("button:has-text('Send')")
        # Button may be briefly disabled while text is processing; force-click it
        send_btn.click(force=True)

    def get_all_messages(self) -> list:
        """
        Return all Dr Khan response bubbles.
        Responses live inside: div.bg-white.text-gray-800 > div.prose
        """
        bubbles = self.page.locator(
            "div.prose.prose-sm"
        ).all_text_contents()
        return [b.strip() for b in bubbles if b.strip()]

    def fill_form_if_present(self, metadata: dict):
        """No mid-conversation forms for Dr Khan — onboarding is done in open_chat()."""
        return False

    def close(self):
        """Close the browser and stop Playwright."""
        self.browser.close()
        self.playwright.stop()

    # ------------------------------------------------------------------
    # Single-turn interface (original behaviour, kept for compatibility)
    # ------------------------------------------------------------------
    def send_prompt(self, prompt: str):
        self.open_chat()
        self.send_message(prompt)
        time.sleep(15)
        msgs = self.get_all_messages()
        return msgs[-1] if msgs else "No response found"
