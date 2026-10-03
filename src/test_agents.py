from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, recall_points
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates

LONG_TURN = (
    "Mình kể thêm một đoạn dài về tin tức và công việc MLOps để thread phình to: "
    "pipeline huấn luyện, logs, prompt snippets, monitoring, chi phí token và độ trễ. " * 3
)


def make_config(tmp_path: Path):
    """Isolated config: state in tmp_path, small compact threshold, always offline."""

    config = load_config()
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        config,
        state_dir=state_dir,
        compact_threshold_tokens=300,
        compact_keep_messages=2,
        live_mode=False,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.file_size("u1") == 0
    assert "## Profile" in store.read_text("u1")

    path = store.write_text("u1", "# User.md: u1\n\n## Profile\n- name: DũngCT\n")
    assert path.exists() and path.name == "User.md"
    assert store.facts("u1") == {"name": "DũngCT"}

    assert store.edit_text("u1", "DũngCT", "DũngCT Stress") is True
    assert store.edit_text("u1", "không tồn tại", "x") is False
    assert store.facts("u1")["name"] == "DũngCT Stress"
    assert store.file_size("u1") > 0


def test_upsert_handles_correction_without_keeping_old_fact(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("u1", "location", "Đà Nẵng")
    store.upsert_fact("u1", "location", "Huế")
    assert store.facts("u1")["location"] == "Huế"
    assert "Đà Nẵng -> Huế" in store.read_text("u1")  # kept only in the change log
    assert store.upsert_fact("u1", "location", "Huế") is False  # idempotent

    store.upsert_fact("u1", "response_style", "ngắn gọn")
    store.upsert_fact("u1", "response_style", "3 bullet; ngắn gọn")
    assert store.facts("u1")["response_style"] == "ngắn gọn; 3 bullet"


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    for _ in range(8):
        agent.reply("u1", "long", LONG_TURN)

    context = agent.compact_memory.context("long")
    assert agent.compaction_count("long") >= 1
    assert len(context["messages"]) <= config.compact_keep_messages + 1
    assert context["summary"]


def test_compact_summary_stays_bounded() -> None:
    manager = CompactMemoryManager(threshold_tokens=200, keep_messages=2, summary_max_items=6)
    for _ in range(30):
        manager.append("t", "user", LONG_TURN)
    assert manager.compaction_count("t") > 3
    assert len(str(manager.context("t")["summary"]).splitlines()) <= 6


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
        "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.",
    ]
    for turn in turns:
        advanced.reply("dungct", "session-1", turn)
        baseline.reply("dungct", "session-1", turn)

    question = "Mình tên gì và hiện tại mình đang ở đâu?"
    # Baseline remembers inside the same thread...
    assert "DũngCT" in baseline.reply("dungct", "session-1", question)["response"]
    # ...but forgets in a new thread, while Advanced recalls the latest fact from User.md.
    assert "DũngCT" not in baseline.reply("dungct", "session-2", question)["response"]
    answer = advanced.reply("dungct", "session-2", question)["response"]
    assert "DũngCT" in answer and "Huế" in answer and "Đà Nẵng" not in answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for _ in range(15):
        advanced.reply("u1", "long", LONG_TURN)
        baseline.reply("u1", "long", LONG_TURN)

    assert advanced.compaction_count("long") >= 1
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.6


def test_questions_and_noise_are_not_stored() -> None:
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Hiện tại mình đang ở đâu?") == {}
    noise = (
        "Có lúc mình đùa rằng hay là chuyển sang product manager. "
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày."
    )
    assert extract_profile_updates(noise) == {}


def test_confidence_threshold_skips_hedged_facts() -> None:
    hedged = "Có lẽ tháng sau mình sẽ chuyển sang Hà Nội."
    assert extract_profile_updates(hedged) == {}
    assert extract_profile_updates(hedged, min_confidence=0.3) == {"location": "Hà Nội"}


def test_stress_dataset_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    conv = load_conversations(config.data_dir / "advanced_long_context.json")[0]
    agent = AdvancedAgent(replace(config, compact_threshold_tokens=1000, compact_keep_messages=4), force_offline=True)
    for turn in conv["turns"]:
        agent.reply(conv["user_id"], "stress", turn)
    assert agent.compaction_count("stress") >= 2
    for item in conv["recall_questions"]:
        answer = agent.reply(conv["user_id"], "fresh", item["question"])["response"]
        assert recall_points(answer, item["expected_contains"]) == 1.0, answer
