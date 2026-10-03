from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import estimate_tokens


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else [data]


def _hits(answer: str, expected: list[str]) -> int:
    low = answer.lower()
    return sum(1 for item in expected if item.lower() in low)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1]: coverage (0.6) + concise (0.2) + no "don't know" (0.2)."""

    if not answer.strip():
        return 0.0
    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    concise = 1.0 if estimate_tokens(answer) <= 120 else 0.5
    knows = 0.0 if "chưa có" in answer.lower() else 1.0
    return round(0.6 * coverage + 0.2 * concise + 0.2 * knows, 3)


def judge_quality(judge_model, question: str, answer: str, expected: list[str]) -> float:
    """LLM-as-judge for live mode; falls back to the heuristic on any failure."""

    prompt = (
        "Chấm điểm câu trả lời từ 0 đến 10 theo độ đúng, đủ ý và ngắn gọn. "
        f"Câu hỏi: {question}\nFact kỳ vọng: {', '.join(expected)}\nCâu trả lời: {answer}\n"
        "Chỉ trả về một số."
    )
    try:
        text = str(judge_model.invoke(prompt).content)
        score = float(re.search(r"\d+(?:\.\d+)?", text).group(0))
        return round(min(max(score / 10, 0.0), 1.0), 3)
    except Exception:
        return heuristic_quality(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Feed every conversation, then ask its recall questions in a fresh thread."""

    judge = None
    if config.live_mode and getattr(agent, "langchain_agent", None) is not None:
        try:
            from model_provider import build_chat_model

            judge = build_chat_model(config.judge_model)
        except Exception:
            judge = None

    user_ids = {conv["user_id"] for conv in conversations}
    start_sizes = {uid: agent.memory_file_size(uid) for uid in user_ids}
    tag = agent_name.lower()
    agent_tokens = prompt_tokens = compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        thread_id = f"{conv['id']}-{tag}"
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], thread_id, turn)
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        for index, item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv['id']}-{tag}-recall-{index}"
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(
                judge_quality(judge, item["question"], answer, expected)
                if judge
                else heuristic_quality(answer, expected)
            )

    growth = sum(agent.memory_file_size(uid) - start_sizes[uid] for uid in user_ids)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
        memory_growth_bytes=growth,
        compactions=compactions,
    )


HEADERS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=HEADERS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
        return "\n".join(lines)


def run_suite(title: str, dataset: Path, config) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    advanced = AdvancedAgent(config)
    for user_id in {conv["user_id"] for conv in conversations}:
        advanced.profile_store.reset(user_id)  # reproducible: start from an empty User.md
    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config), conversations, config),
        run_agent_benchmark("Advanced", advanced, conversations, config),
    ]
    print(f"\n## {title} ({dataset.name}, {len(conversations)} conversation(s))\n")
    print(format_rows(rows))
    return rows


def main() -> None:
    """Run the standard benchmark and the long-context stress benchmark."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config(Path(__file__).resolve().parent.parent)
    mode = f"live ({config.model.provider}/{config.model.model_name})" if config.live_mode else "offline"
    print(
        f"Mode: {mode} | compact threshold={config.compact_threshold_tokens} tokens, "
        f"keep={config.compact_keep_messages} messages"
    )
    run_suite("Standard Benchmark", config.data_dir / "conversations.json", config)
    run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config)


if __name__ == "__main__":
    main()
