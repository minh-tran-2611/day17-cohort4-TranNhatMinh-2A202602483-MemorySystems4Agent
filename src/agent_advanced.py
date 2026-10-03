from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from agent_baseline import _message_text
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    estimate_tokens,
    extract_profile_updates,
    is_question,
)
from model_provider import build_chat_model

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn. Dưới đây là hồ sơ User.md của người dùng "
    "(fact mới nhất là đúng, Change log chỉ là lịch sử) và tóm tắt hội thoại cũ. "
    "Dùng chúng để trả lời đúng style người dùng thích. Khi người dùng cung cấp fact ổn định "
    "mới (tên, nơi ở, nghề, sở thích, style), hãy gọi tool update_user_fact."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent `User.md` + compact memory.

    Per turn: extract facts -> upsert User.md -> append to compact memory ->
    prompt = system + User.md + summary + recent messages -> answer.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self._active_user_id: str | None = None

        self.langchain_agent = None
        if self.config.live_mode and not force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception as exc:
                warnings.warn(f"AdvancedAgent falling back to offline mode: {exc}")

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        return self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # -- shared steps -------------------------------------------------------

    def _remember(self, user_id: str, thread_id: str, message: str) -> tuple[dict[str, str], int]:
        """Steps 1-4: persist stable facts, append to short-term memory, charge the prompt."""

        updates = extract_profile_updates(message, self.config.profile_confidence_threshold)
        changed = self.profile_store.apply_updates(user_id, updates)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return changed, prompt_tokens

    def _finish(self, thread_id: str, message: str, answer: str) -> None:
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + estimate_tokens(message) + estimate_tokens(answer)
        )

    # -- offline path -------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changed, prompt_tokens = self._remember(user_id, thread_id, message)
        answer = self._offline_response(user_id, thread_id, message)
        if changed and not is_question(message):
            answer += " Đã cập nhật User.md: " + ", ".join(f"{k}={v}" for k, v in changed.items()) + "."
        self._finish(thread_id, message, answer)
        return {
            "response": answer,
            "prompt_tokens": prompt_tokens,
            "profile_updates": changed,
            "mode": "offline",
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: system + User.md + compact summary + kept messages."""

        context = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(str(context["summary"]))
            + sum(estimate_tokens(m["content"]) for m in context["messages"])  # type: ignore[union-attr]
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Deterministic answer: questions are answered from persisted User.md facts."""

        if is_question(message):
            return answer_from_facts(message, self.profile_store.facts(user_id))
        return "Đã ghi nhận."

    # -- live path ----------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changed, prompt_tokens = self._remember(user_id, thread_id, message)
        self._active_user_id = user_id
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id))),
        )
        answer = _message_text(result["messages"][-1])
        self._finish(thread_id, message, answer)
        return {"response": answer, "prompt_tokens": prompt_tokens, "profile_updates": changed, "mode": "live"}

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + InMemorySaver + User.md tools + dynamic prompt
        injecting the profile + summarization middleware for long threads."""

        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
        from langchain.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        store = self.profile_store
        model = build_chat_model(self.config.model)

        def current_user() -> str:
            return self._active_user_id or "anonymous"

        @tool
        def read_user_memory() -> str:
            """Đọc toàn bộ User.md của người dùng hiện tại."""
            return store.read_text(current_user())

        @tool
        def update_user_fact(key: str, value: str) -> str:
            """Ghi/cập nhật một fact ổn định vào User.md. key thuộc: name, location, profession,
            response_style, interests, favorite_drink, favorite_food, pet."""
            changed = store.upsert_fact(current_user(), key, value)
            return "updated" if changed else "unchanged"

        compact = self.compact_memory

        @dynamic_prompt
        def inject_memory(request: ModelRequest) -> str:
            ctx = getattr(request.runtime, "context", None)
            user_id = getattr(ctx, "user_id", None) or current_user()
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n{store.read_text(user_id)}"

        return create_agent(
            model,
            tools=[read_user_memory, update_user_fact],
            middleware=[
                inject_memory,
                SummarizationMiddleware(
                    model,
                    trigger=("tokens", compact.threshold_tokens),
                    keep=("messages", compact.keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
