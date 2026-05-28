"""
chat/chat_format.py
===================
Conversation formatting and history management for the GPT LLM.

CHAT FORMAT
===========
Conversations are formatted as:

    User: Hello, how are you?
    Assistant: I'm doing well, thanks for asking!
    User: What's the capital of France?
    Assistant:

The model is prompted with the full conversation history so it can
produce contextually relevant responses.

CONTEXT WINDOW MANAGEMENT
==========================
Transformers have a fixed context window (max_seq_len).
As conversation grows, we must manage memory:

Strategy 1 — Sliding Window (implemented here):
    Keep the most recent N tokens that fit in the context window.
    Old turns are dropped from the beginning.
    Simple and effective for most use cases.

Strategy 2 — Summarization:
    Summarize old turns and prepend a summary.
    Better recall but requires a summarization model.

The system prompt (if provided) is always kept — only user/assistant
turns are truncated.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# MESSAGE TYPES
# ============================================================

class Message:
    """Represents a single conversation turn."""

    ROLES = ("user", "assistant", "system")

    def __init__(self, role: str, content: str):
        assert role in self.ROLES, f"Role must be one of {self.ROLES}"
        self.role = role
        self.content = content.strip()

    def format(self) -> str:
        """Format as 'Role: content' string."""
        role_display = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }
        return f"{role_display[self.role]}: {self.content}"

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, content={self.content[:50]!r})"


# ============================================================
# CONVERSATION MANAGER
# ============================================================

class ConversationManager:
    """
    Manages a multi-turn conversation for the GPT chatbot.

    Handles:
    - Adding user/assistant messages
    - Formatting conversation history as a prompt
    - Context window management (truncation)
    - System prompt handling

    Usage:
        conv = ConversationManager(system_prompt="You are a helpful AI.")
        conv.add_user("What is 2+2?")
        prompt = conv.build_prompt()  # pass to model
        conv.add_assistant("2+2 = 4!")

    Args:
        system_prompt : Optional system message shown at start
        tokenizer     : LLMTokenizer (for token counting)
        max_seq_len   : Maximum context window tokens
        max_turns     : Maximum number of turns to keep (None = no limit)
    """

    # Template strings for formatting
    USER_PREFIX = "User: "
    ASSISTANT_PREFIX = "Assistant: "
    SYSTEM_PREFIX = "System: "
    TURN_SEP = "\n"           # Separator between turns
    PROMPT_END = "\nAssistant:"  # What we append to prompt the model

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        tokenizer=None,
        max_seq_len: int = 512,
        max_turns: int = 20,
    ):
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_turns = max_turns
        self.history: List[Message] = []

    def add_user(self, text: str):
        """Add a user message to the history."""
        self.history.append(Message("user", text))

    def add_assistant(self, text: str):
        """Add an assistant response to the history."""
        self.history.append(Message("assistant", text))

    def add_system(self, text: str):
        """Add a system message (overrides system_prompt)."""
        self.system_prompt = text

    def clear(self):
        """Clear conversation history (but keep system prompt)."""
        self.history = []

    def build_prompt(self) -> str:
        """
        Build the full prompt string from conversation history.

        Format:
            [System: <system_prompt>\\n]
            User: <turn1>\\n
            Assistant: <response1>\\n
            User: <turn2>\\n
            Assistant: <response2>\\n
            User: <latest>\\n
            Assistant:   ← model continues from here

        Context management:
            If the prompt would exceed max_seq_len tokens, we drop the
            oldest non-system turns until it fits.

        Returns:
            Formatted prompt string ready for tokenization
        """
        parts = []

        # Add system prompt
        if self.system_prompt:
            parts.append(f"{self.SYSTEM_PREFIX}{self.system_prompt}")

        # Add conversation history
        # Limit to last max_turns turns
        history = self.history[-self.max_turns:] if self.max_turns else self.history
        for msg in history:
            if msg.role == "user":
                parts.append(f"{self.USER_PREFIX}{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"{self.ASSISTANT_PREFIX}{msg.content}")

        # Join all parts
        prompt = self.TURN_SEP.join(parts)

        # Add prompt for assistant to continue
        if not prompt.endswith(self.PROMPT_END):
            prompt += self.PROMPT_END

        # Context window truncation
        if self.tokenizer is not None:
            prompt = self._truncate_to_fit(prompt)

        return prompt

    def _truncate_to_fit(self, prompt: str) -> str:
        """
        Truncate the prompt to fit within max_seq_len tokens.

        Drops oldest turns first, preserving:
        - System prompt
        - At least the last user message + assistant prompt

        Args:
            prompt: Full prompt string

        Returns:
            Truncated prompt string
        """
        # Check if it fits
        token_count = self.tokenizer.count_tokens(prompt)
        if token_count <= self.max_seq_len - 100:  # 100 token buffer for generation
            return prompt

        # Gradually drop oldest turns
        history = list(self.history)
        while len(history) > 2:  # Keep at least last user + empty assistant
            history = history[1:]  # Drop oldest turn

            # Rebuild with reduced history
            parts = []
            if self.system_prompt:
                parts.append(f"{self.SYSTEM_PREFIX}{self.system_prompt}")
            for msg in history:
                if msg.role == "user":
                    parts.append(f"{self.USER_PREFIX}{msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"{self.ASSISTANT_PREFIX}{msg.content}")
            candidate = self.TURN_SEP.join(parts) + self.PROMPT_END

            token_count = self.tokenizer.count_tokens(candidate)
            if token_count <= self.max_seq_len - 100:
                return candidate

        return prompt  # Return as-is if we can't truncate further

    def extract_response(self, full_text: str) -> str:
        """
        Extract the assistant's response from the model's output.

        The model generates text continuing from "Assistant:",
        so we extract everything after the last "Assistant:" marker.

        Args:
            full_text: Full generated text (prompt + response)

        Returns:
            Just the assistant's response text
        """
        # Find the last "Assistant:" marker
        prefix = self.PROMPT_END.strip()
        last_idx = full_text.rfind(prefix)
        if last_idx == -1:
            # Fallback: return the full text
            return full_text.strip()

        # Extract everything after "Assistant:"
        response = full_text[last_idx + len(prefix):].strip()

        # Trim at the next "User:" marker if present (in case model over-generates)
        if "\nUser:" in response:
            response = response[:response.index("\nUser:")].strip()

        return response

    def get_context_length(self) -> int:
        """Return token count of current prompt."""
        if self.tokenizer is None:
            return len(self.build_prompt()) // 4  # rough estimate
        return self.tokenizer.count_tokens(self.build_prompt())

    def __repr__(self) -> str:
        return (
            f"ConversationManager("
            f"turns={len(self.history)}, "
            f"context≈{self.get_context_length()} tokens)"
        )
