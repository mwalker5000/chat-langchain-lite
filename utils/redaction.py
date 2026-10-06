"""Input-side redaction of pasted credentials and PII.

Nothing upstream of the agent strips secrets, so a credential pasted into a
question would otherwise reach the model and be persisted verbatim in agent
state and the trace store. Redaction happens on the way IN, before either.
"""

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

_API_KEY = "[REDACTED_API_KEY]"
_SSN = "[REDACTED_SSN]"

# Type-specific placeholders, so the model can still reason about the fact that
# a credential was present without seeing its value.
PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"lsv2_[A-Za-z0-9_]{8,}"), _API_KEY),
    (re.compile(r"sk-[A-Za-z0-9-]{16,}"), _API_KEY),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), _API_KEY),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), _API_KEY),
    (re.compile(r"AKIA[0-9A-Z]{12,}"), _API_KEY),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), _SSN),
)


def redact(text: str) -> str:
    """Replace every known secret or PII shape in `text` with its placeholder."""
    for pattern, placeholder in PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def redact_content(content: Any) -> Any:
    """Redact a message `content` value, handling both strings and content blocks."""
    if isinstance(content, str):
        return redact(content)
    if isinstance(content, list):
        return [
            {**block, "text": redact(block["text"])}
            if isinstance(block, dict) and isinstance(block.get("text"), str)
            else redact_content(block)
            for block in content
        ]
    return content


class SecretRedactionMiddleware(AgentMiddleware):
    """Rewrite human message content so pasted secrets never reach the model."""

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        # Replacement works by id: the `messages` reducer upserts, and it has
        # already stamped an id on every message in state by this point.
        redacted = []
        for message in state["messages"]:
            if not isinstance(message, HumanMessage):
                continue
            content = redact_content(message.content)
            if content != message.content:
                redacted.append(message.model_copy(update={"content": content}))
        return {"messages": redacted} if redacted else None
