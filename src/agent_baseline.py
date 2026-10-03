from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import answer_from_facts, estimate_tokens, extract_profile_updates, is_question
from model_provider import build_chat_model

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Chỉ dùng thông tin có trong cuộc trò chuyện hiện tại. "
    "Nếu không biết thì nói chưa có thông tin. Trả lời ngắn gọn."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Keeps the full message list per `thread_id` (short-term memory).
    - No persistent `User.md`, no compaction.
    - A new thread starts empty, so long-term facts are forgotten by design.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if self.config.live_mode and not force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception as exc:  # missing SDK / bad key -> stay usable offline
                warnings.warn(f"BaselineAgent falling back to offline mode: {exc}")

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        return self._reply_live(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    # -- shared accounting --------------------------------------------------

    def _record_turn(self, session: SessionState, message: str) -> int:
        """Append the user turn and charge the prompt: system + whole history + message."""

        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(
            estimate_tokens(m["content"]) for m in session.messages
        )
        session.prompt_tokens_processed += prompt_tokens
        return prompt_tokens

    def _record_answer(self, session: SessionState, message: str, answer: str) -> None:
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(message) + estimate_tokens(answer)

    # -- offline path -------------------------------------------------------

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        prompt_tokens = self._record_turn(session, message)

        if is_question(message):
            # Only facts said earlier in *this* thread are visible to the baseline.
            thread_facts: dict[str, str] = {}
            for past in session.messages[:-1]:
                if past["role"] == "user":
                    thread_facts.update(extract_profile_updates(past["content"]))
            answer = answer_from_facts(message, thread_facts)
        else:
            answer = "Đã ghi nhận."

        self._record_answer(session, message, answer)
        return {"response": answer, "prompt_tokens": prompt_tokens, "mode": "offline"}

    # -- live path ----------------------------------------------------------

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        prompt_tokens = self._record_turn(session, message)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = _message_text(result["messages"][-1])
        self._record_answer(session, message, answer)
        return {"response": answer, "prompt_tokens": prompt_tokens, "mode": "live"}

    def _maybe_build_langchain_agent(self):
        """Live agent: `create_agent` + `InMemorySaver` (thread-scoped memory only)."""

        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        return create_agent(
            build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)
