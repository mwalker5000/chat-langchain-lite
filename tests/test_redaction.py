import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from utils.redaction import SecretRedactionMiddleware, redact

API_KEY = "[REDACTED_API_KEY]"
SSN = "[REDACTED_SSN]"

# Fixtures are assembled from prefix + body at runtime so this file never holds
# a literal string that credential scanners flag as a live secret.
LANGSMITH_KEY = "lsv2_" + "pt_sk_7fQ9m2Vx8Kp4Nz6Rw3Tc1Hd5"
GITHUB_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8J9k0L1m2N3o4P5"

SECRETS = [
    (LANGSMITH_KEY, API_KEY),
    ("sk-" + "proj-A1b2C3d4E5f6G7h8J9k0", API_KEY),
    ("sk-ant-" + "api03-A1b2C3d4E5f6G7h8J9k0", API_KEY),
    (GITHUB_TOKEN, API_KEY),
    ("xoxb-" + "1234567890-abcdefGHIJ", API_KEY),
    ("AKIA" + "IOSFODNN7EXAMPLE", API_KEY),
    ("444-33-2235", SSN),
]


@pytest.mark.parametrize("secret,placeholder", SECRETS)
def test_each_pattern_is_redacted(secret, placeholder):
    result = redact(f"here it is: {secret} — please help")
    assert secret not in result
    assert placeholder in result


@pytest.mark.parametrize(
    "text",
    [
        "call me on 555-1234 or 1-800-555-0199",
        "the release window is 2026-09-29 to 2026-09-30",
        "error codes 12-345-6789 and 4-56-7890 are unrelated",
        "pip install langchain-core",
    ],
)
def test_non_secret_text_is_untouched(text):
    assert redact(text) == text


def test_content_blocks_are_redacted():
    message = HumanMessage(id="1", content=[{"type": "text", "text": f"key {LANGSMITH_KEY}"}])
    update = SecretRedactionMiddleware().before_model({"messages": [message]}, None)
    assert update["messages"][0].content == [{"type": "text", "text": f"key {API_KEY}"}]


def test_redacted_message_does_not_reach_state_or_model():
    seen = []

    class RecordingModel(GenericFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.extend(str(m.content) for m in messages)
            return super()._generate(messages, stop, run_manager, **kwargs)

    agent = create_agent(
        model=RecordingModel(messages=iter([AIMessage(content="rotate it")])),
        tools=[],
        middleware=[SecretRedactionMiddleware()],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": f"why 401? {LANGSMITH_KEY}"}]})

    assert all(LANGSMITH_KEY not in content for content in seen)
    assert all(LANGSMITH_KEY not in str(m.content) for m in result["messages"])
    assert API_KEY in result["messages"][0].content


def test_every_human_message_is_redacted_not_just_the_first():
    state = {
        "messages": [
            HumanMessage(id="1", content=f"first {GITHUB_TOKEN}"),
            AIMessage(id="2", content="ok"),
            HumanMessage(id="3", content="second 444-33-2235"),
        ]
    }
    update = SecretRedactionMiddleware().before_model(state, None)
    assert [m.id for m in update["messages"]] == ["1", "3"]
    assert update["messages"][1].content == f"second {SSN}"
