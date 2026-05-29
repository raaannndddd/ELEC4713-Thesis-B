import time
from web_automation.base_web_client import BaseWebClient


class DoctronicClient(BaseWebClient):

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def open_chat(self):
        self._terms_agreed = False
        self._current_metadata = {}
        self.page.goto("https://www.doctronic.ai")
        time.sleep(5)

    # ------------------------------------------------------------------
    # TOS helper (called from send_message so ConvManager sees messages)
    # ------------------------------------------------------------------
    def _accept_tos_if_present(self) -> bool:
        """Click the TOS checkbox if it is visible. Returns True if clicked."""
        if self._terms_agreed:
            return False
        selectors = [
            "input[data-react-aria-pressable][type='checkbox']",
            "label[data-test-id='tos-checkbox']",
            "input[name='agree-terms-check']",
        ]
        for sel in selectors:
            try:
                tos = self.page.locator(sel)
                if tos.count() and tos.first.is_visible():
                    tos.first.click(force=True)
                    self._terms_agreed = True
                    print("[FORM] Accepted Terms of Service")
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Multi-turn conversation interface
    # ------------------------------------------------------------------
    def send_message(self, text: str):
        """Type, submit a message, then immediately accept TOS if present."""
        # Wait for the chat textarea to become enabled (disabled while Doctronic generates)
        try:
            self.page.wait_for_selector(
                "textarea#chat-input:not([disabled])",
                timeout=60000,
            )
        except Exception:
            # Timed out — if the textarea is gone entirely, the consultation ended
            if not self.page.locator("textarea#chat-input").count():
                print("[DOCTRONIC] Chat ended (Clinical Report shown) — skipping send.")
                return

        # Only target the named chat input; never fall through to generic textarea
        ta = self.page.locator("textarea#chat-input, textarea[name='chat-input']")
        if not ta.count():
            print("[DOCTRONIC] Chat textarea not found — consultation has ended.")
            return

        ta.first.fill(text)
        time.sleep(0.5)
        try:
            btn = self.page.locator("button[aria-label='Send message']")
            btn.first.click()
        except Exception:
            self.page.keyboard.press("Enter")
        time.sleep(3)
        self._accept_tos_if_present()

    def get_all_messages(self) -> list:
        """Return all chatbot response bubbles visible on the current page."""
        # Wait for Doctronic to finish generating (textarea is disabled while it types)
        try:
            self.page.wait_for_selector(
                "textarea#chat-input:not([disabled])",
                timeout=45000,
            )
        except Exception:
            pass

        # Click a neutral spot to dismiss any Doctronic timeout / safety overlay
        try:
            self.page.mouse.click(400, 100)
            time.sleep(0.5)
        except Exception:
            pass

        selectors = [
            "[class*='Markdown-module__'] p",
            "[class*='Markdown-module__']",
            "[class*='AssistantMessage-module'] p",
            "[class*='AssistantMessage'] p",
            ".message-text-content",
        ]
        for sel in selectors:
            try:
                els = self.page.locator(sel).all_text_contents()
                clean = [t.strip() for t in els if t.strip()]
                if clean:
                    return clean
            except Exception:
                continue
        return []

    def fill_form_if_present(self, metadata: dict) -> bool:
        """
        Fill the age/sex form if it is currently visible.

        TOS is handled inside send_message(), so this method only deals
        with the AgeSexForm fieldset that appears after Doctronic's first reply.

        Returns True if the form was found and submitted (caller should wait
        for the chatbot's clinical follow-up before reading messages).
        """
        age_input = self.page.locator("input[name='age']")
        if not (age_input.count() and age_input.first.is_visible()):
            return False

        age = str(metadata.get("age", ""))
        gender = metadata.get("gender", "").lower()

        # Fill age
        age_input.first.click()
        age_input.first.fill(age)
        time.sleep(0.3)

        # Select sex radio — Female is pre-selected; only click Male if needed
        if gender == "male":
            try:
                self.page.locator(
                    "label[class*='AgeSexForm-module__'][class*='sexRadio']"
                ).filter(has_text="Male").first.click()
            except Exception:
                try:
                    self.page.get_by_text("Male", exact=True).first.click()
                except Exception:
                    pass
        time.sleep(0.3)

        # Submit
        submit = self.page.locator("button[data-test-id='age-sex-form-submit']")
        if submit.count():
            submit.first.click()
            print(f"[FORM] Submitted age={age}, sex={'Male' if gender == 'male' else 'Female'}")
            return True

        return False

    def get_clinical_report(self) -> str:
        """
        Extract Doctronic's final Clinical Report card if it is present.

        Expands the collapsed Assessment & Plan and SOAP Note sections,
        then returns all report text joined as a single string.
        Returns an empty string if the report has not appeared yet.
        """
        container = self.page.locator("div[data-message-role='summary']")
        if not container.count():
            return ""

        parts = []

        # Top-level summary paragraph
        try:
            summary_el = container.locator(".Markdown-module__pA1MkG__markdown").first
            text = summary_el.text_content()
            if text and text.strip():
                parts.append(text.strip())
        except Exception:
            pass

        # Expand and extract each collapsible section (Assessment & Plan, SOAP Note)
        try:
            details_els = container.locator("details")
            for i in range(details_els.count()):
                try:
                    details_els.nth(i).locator("summary").click()
                    time.sleep(0.5)
                    title_el = details_els.nth(i).locator("p[class*='ui-heading']").first
                    title = title_el.text_content().strip() if title_el.count() else f"Section {i + 1}"
                    content_el = details_els.nth(i).locator(".Markdown-module__pA1MkG__markdown").first
                    content = content_el.text_content().strip() if content_el.count() else ""
                    if content:
                        parts.append(f"=== {title} ===\n{content}")
                except Exception:
                    pass
        except Exception:
            pass

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Single-turn interface (used by run_prompts.py / run_all_chatbots_v2)
    # ------------------------------------------------------------------
    def _get_assistant_responses(self) -> list:
        return self.get_all_messages()

    def send_prompt(self, prompt):
        self.open_chat()

        # Submit the initial prompt
        textarea = self.page.locator("textarea").first
        textarea.fill(prompt)
        time.sleep(0.5)
        try:
            send_btn = self.page.locator(
                "button[type='submit'], button[aria-label*='Send'], button[aria-label*='send']"
            )
            if send_btn.count():
                send_btn.first.click()
            else:
                self.page.keyboard.press("Enter")
        except Exception:
            self.page.keyboard.press("Enter")

        # Accept TOS (appears before Doctronic's first reply)
        time.sleep(3)
        self._accept_tos_if_present()

        # Wait for Doctronic to ask for age/sex, then fill the form
        time.sleep(10)
        meta = getattr(self, "_current_metadata", {})
        if meta:
            filled = self.fill_form_if_present(meta)
            if filled:
                time.sleep(15)
        else:
            time.sleep(15)

        messages = self.get_all_messages()
        print(f"[RESPONSES] Found {len(messages)} assistant messages")
        return messages[-1] if messages else "No response found"
