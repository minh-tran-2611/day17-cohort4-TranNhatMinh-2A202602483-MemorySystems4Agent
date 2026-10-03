from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~4 characters per token, 0 for empty text."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


# ---------------------------------------------------------------------------
# Persistent profile: User.md
# ---------------------------------------------------------------------------

FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời",
    "interests": "Mối quan tâm",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
}

# Facts that accumulate (set union) instead of being overwritten by a correction.
MERGE_KEYS = {"response_style", "interests"}
MAX_MERGED_ITEMS = 8
MAX_CHANGELOG_LINES = 10

_FACT_LINE = re.compile(r"^- (\w+): (.+)$")


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`, one markdown file per user.

    Layout:
        # User.md: <user_id>
        ## Profile
        - key: value
        ## Change log
        - key: old -> new
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip()).strip("._") or "anonymous"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return f"# User.md: {user_id}\n\n## Profile\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def reset(self, user_id: str) -> None:
        path = self.path_for(user_id)
        if path.exists():
            path.unlink()

    # -- structured helpers -------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        in_profile = False
        for line in self.read_text(user_id).splitlines():
            if line.startswith("## "):
                in_profile = line.strip() == "## Profile"
                continue
            match = _FACT_LINE.match(line.strip())
            if in_profile and match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def _changelog(self, user_id: str) -> list[str]:
        lines, in_log = [], False
        for line in self.read_text(user_id).splitlines():
            if line.startswith("## "):
                in_log = line.strip() == "## Change log"
                continue
            if in_log and line.startswith("- "):
                lines.append(line)
        return lines

    def _render(self, user_id: str, facts: dict[str, str], changelog: list[str]) -> str:
        ordered = [k for k in FACT_LABELS if k in facts] + [k for k in facts if k not in FACT_LABELS]
        parts = [f"# User.md: {user_id}", "", "## Profile"]
        parts += [f"- {key}: {facts[key]}" for key in ordered]
        if changelog:
            parts += ["", "## Change log"] + changelog[-MAX_CHANGELOG_LINES:]
        return "\n".join(parts) + "\n"

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or update one fact. Returns True when User.md changed.

        Conflict handling: a scalar fact is overwritten by the newer value and the old
        value moves to the change log, so the profile never holds two contradicting facts.
        Merge keys (style, interests) accumulate unique items, capped to stay small.
        """

        facts = self.facts(user_id)
        changelog = self._changelog(user_id)
        old = facts.get(key)
        if key in MERGE_KEYS and old:
            items = [i.strip() for i in old.split(";") if i.strip()]
            for item in (i.strip() for i in value.split(";")):
                if item and item.lower() not in {x.lower() for x in items}:
                    items.append(item)
            new = "; ".join(items[-MAX_MERGED_ITEMS:])
        else:
            new = value.strip()
        if old == new:
            return False
        facts[key] = new
        if old and key not in MERGE_KEYS:
            changelog.append(f"- {key}: {old} -> {new}")
        self.write_text(user_id, self._render(user_id, facts, changelog))
        return True

    def apply_updates(self, user_id: str, updates: dict[str, str]) -> dict[str, str]:
        """Upsert many facts; return only the ones that actually changed the file."""

        changed = {}
        for key, value in updates.items():
            if self.upsert_fact(user_id, key, value):
                changed[key] = value
        return changed


# ---------------------------------------------------------------------------
# Fact extraction (rule-based, Vietnamese)
# ---------------------------------------------------------------------------

CITIES = [
    "Hồ Chí Minh", "Sài Gòn", "Đà Nẵng", "Hà Nội", "Huế", "Hải Phòng",
    "Cần Thơ", "Nha Trang", "Đà Lạt", "Quy Nhơn", "Vũng Tàu",
]
_CITY_RE = re.compile("|".join(re.escape(c) for c in sorted(CITIES, key=len, reverse=True)))
_LOCATION_CUE = re.compile(
    r"(?:(?:mình|tôi|hiện|đang|vẫn|giờ|sống|làm việc|chuyển)\s+(?:đang\s+)?ở"
    r"|sang|chuyển (?:về|tới|đến)"
    r"|nơi ở(?: hiện tại)?(?: của mình)? là)\s*$"
)
_LOCATION_NOISE = ("chỉ là nơi", "không phải nơi ở", "đừng lấy", "ví dụ cũ", "nếu ai đó", "đi họp", "bay ra họp")

_ROLE_RE = re.compile(
    r"\b((?:backend|frontend|full[- ]?stack|mlops|ml|data|ai|software|devops|platform|qa)\s+engineer"
    r"|data scientist|product manager|project manager|designer|developer)\b",
    re.IGNORECASE,
)
_ROLE_CUE = re.compile(r"(?:làm|là|sang|nghề|vai trò)\s+(?:một\s+)?$")
_NEGATION = ("không", "đừng", "chứ", "thôi")
_JOKE_OR_HYPOTHETICAL = ("đùa", "giả sử", "nếu ai đó", "ví dụ như")

_HEDGES = ("có lẽ", "hình như", "chắc là", "đang cân nhắc", "dự định", "có thể sẽ", "định chuyển")
_CORRECTION_CUES = ("đính chính", "thực ra", "cập nhật", "không còn", "giờ", "hiện tại", "vẫn", "nhắc lại")

_INTEREST_KEYWORDS = [
    "async Python", "Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "LangGraph",
    "benchmark memory", "memory architecture", "chạy bộ", "lo-fi",
]

_QUESTION_MARKERS = (" gì", "ở đâu", "là ai", "thế nào", "bao nhiêu")
_REQUEST_PREFIXES = (
    "nhắc lại giúp", "nhắc lại cho", "hãy nhắc", "tóm tắt", "bạn có biết",
    "bạn có thể nhắc", "bạn thử nhớ", "bạn biết",
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if s.strip()]


def is_question(sentence: str) -> bool:
    """True for questions / recall requests, which must never be stored as facts."""

    low = sentence.strip().lower()
    if low.endswith("?"):
        return True
    if any(marker in low for marker in _QUESTION_MARKERS):
        return True
    # "Nhắc lại style..." is a request; "Nhắc lại lần cuối: tên X..." is a restated fact.
    if re.match(r"^(?:hãy |bạn )?(?:nhắc lại|kể lại|liệt kê)(?! lần)", low):
        return True
    return any(prefix in low for prefix in _REQUEST_PREFIXES)


def _confidence(sentence: str, base: float = 0.8) -> float:
    low = sentence.lower()
    if any(h in low for h in _HEDGES):
        return 0.4
    if any(c in low for c in _CORRECTION_CUES):
        return 0.95
    return base


def _extract_name(sentence: str) -> str | None:
    match = re.search(r"tên (?:mình |tôi |của mình )?là\s+(.+)", sentence, re.IGNORECASE)
    if not match:
        return None
    words = []
    for raw in match.group(1).split():
        word = raw.strip(",.!?;:\"'()")
        if not word or not word[0].isupper():
            break
        words.append(word)
        if raw[-1] in ",.!?;:":
            break
    return " ".join(words) or None


def _extract_location(sentence: str) -> str | None:
    low = sentence.lower()
    if any(noise in low for noise in _LOCATION_NOISE):
        return None
    found = None
    for match in _CITY_RE.finditer(sentence):
        before = sentence[: match.start()].lower()
        window = before[-15:]
        if _LOCATION_CUE.search(before) and not any(neg in window for neg in _NEGATION):
            found = match.group(0)  # the last positive mention wins (corrections come later)
    return found


def _extract_profession(sentence: str) -> str | None:
    low = sentence.lower()
    if any(j in low for j in _JOKE_OR_HYPOTHETICAL):
        return None
    found = None
    for match in _ROLE_RE.finditer(sentence):
        before = sentence[: match.start()].lower()
        window = before[-16:]
        if _ROLE_CUE.search(before) and not any(neg in window for neg in _NEGATION):
            found = match.group(1)
    if found and found.lower().startswith("mlops"):
        found = "MLOps" + found[5:]
    return found


def _extract_style(sentence: str) -> str | None:
    low = sentence.lower()
    if not any(cue in low for cue in ("trả lời", "giải thích", "style", "trình bày")):
        return None
    features = []
    if "ngắn" in low or "gọn" in low:
        features.append("ngắn gọn")
    bullet = re.search(r"(\d+)\s+bullet", low)
    if bullet:
        features.append(f"{bullet.group(1)} bullet")
    elif "bullet" in low:
        features.append("dạng bullet")
    if "ví dụ" in low:
        features.append("có ví dụ thực chiến" if "thực chiến" in low else "có ví dụ thực tế")
    if "trade-off" in low:
        features.append("nhấn trade-off")
    if "cấu trúc" in low:
        features.append("có cấu trúc")
    return "; ".join(features) or None


def _extract_interests(sentence: str) -> str | None:
    low = sentence.lower()
    if not ("thích" in low or "quan tâm" in low) or "không thích" in low:
        return None
    found, consumed = [], low
    for keyword in _INTEREST_KEYWORDS:
        if keyword.lower() in consumed:
            found.append(keyword)
            consumed = consumed.replace(keyword.lower(), " ")
    return "; ".join(found) or None


def _extract_simple(pattern: str, sentence: str) -> str | None:
    match = re.search(pattern, sentence, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_profile_candidates(message: str) -> dict[str, tuple[str, float]]:
    """Return {fact_key: (value, confidence)} found in a user message.

    Question / request sentences are skipped so "Mình tên gì?" never becomes a fact.
    """

    candidates: dict[str, tuple[str, float]] = {}
    for sentence in split_sentences(message):
        if is_question(sentence):
            continue
        conf = _confidence(sentence)
        extracted = {
            "name": _extract_name(sentence),
            "location": _extract_location(sentence),
            "profession": _extract_profession(sentence),
            "response_style": _extract_style(sentence),
            "interests": _extract_interests(sentence),
            "favorite_drink": _extract_simple(r"đồ uống yêu thích (?:của mình )?là\s+([^.,!?;]+)", sentence),
            "favorite_food": _extract_simple(r"món (?:ăn )?(?:yêu thích|ruột) (?:của mình )?là\s+([^.,!?;]+)", sentence),
        }
        pet = re.search(r"nuôi (?:một |1 )?(?:bé |con )?([\w-]+)(?: tên ([A-ZĐ]\w*))?", sentence)
        if pet:
            extracted["pet"] = pet.group(1) + (f" tên {pet.group(2)}" if pet.group(2) else "")
        for key, value in extracted.items():
            if not value:
                continue
            if key in MERGE_KEYS and key in candidates:
                value = f"{candidates[key][0]}; {value}"
            candidates[key] = (value, conf)
    return candidates


def extract_profile_updates(message: str, min_confidence: float = 0.6) -> dict[str, str]:
    """Convert raw user text into stable profile facts that pass the confidence threshold."""

    return {
        key: value
        for key, (value, conf) in extract_profile_candidates(message).items()
        if conf >= min_confidence
    }


# ---------------------------------------------------------------------------
# Answering from a fact dict (shared by both agents so the comparison is fair)
# ---------------------------------------------------------------------------

_QUESTION_TOPICS = [
    (("tên", "là ai"), ["name"]),
    (("ở đâu", "nơi ở", "còn ở"), ["location"]),
    (("nghề", "làm gì", "công việc"), ["profession"]),
    (("style", "kiểu trả lời", "trả lời"), ["response_style"]),
    (("đồ uống", "uống gì"), ["favorite_drink"]),
    (("món ăn",), ["favorite_food"]),
    (("nuôi", "con gì", "thú cưng"), ["pet"]),
    (("quan tâm", "kỹ thuật"), ["interests"]),
    (("là ai", "tóm tắt", "mô tả"), ["name", "profession", "interests"]),
]


def requested_fact_keys(question: str) -> list[str]:
    low = question.lower()
    keys: list[str] = []
    for markers, fact_keys in _QUESTION_TOPICS:
        if any(m in low for m in markers):
            keys += [k for k in fact_keys if k not in keys]
    return keys


def answer_from_facts(question: str, facts: dict[str, str]) -> str:
    keys = requested_fact_keys(question) or list(facts)
    if not facts or not keys:
        return "Mình chưa có thông tin này trong bộ nhớ hiện tại."
    lines = [f"- {FACT_LABELS.get(k, k)}: {facts.get(k, 'chưa có trong bộ nhớ')}" for k in keys]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: one short bullet per older message, keeping the latest items.

    A previous summary can be passed as a message with role "summary"; its bullets are
    carried over so repeated compactions stay bounded at `max_items` lines.
    """

    items: list[str] = []
    for message in messages:
        role, content = message.get("role", "user"), message.get("content", "")
        if role == "summary":
            items += [line[2:] for line in content.splitlines() if line.startswith("- ")]
            continue
        if role != "user":
            continue  # assistant turns are derivable from user turns in this lab
        facts = extract_profile_updates(content)
        if facts:
            items.append("facts: " + ", ".join(f"{k}={v}" for k, v in facts.items()))
        sentences = split_sentences(content)
        if sentences:
            snippet = sentences[0]
            items.append("user: " + (snippet[:117] + "..." if len(snippet) > 120 else snippet))
    return "\n".join(f"- {item}" for item in items[-max_items:])


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Keeps the latest `keep_messages` messages verbatim; when the thread (summary +
    messages) exceeds `threshold_tokens`, older messages are folded into a summary.
    """

    threshold_tokens: int
    keep_messages: int
    summary_max_items: int = 6
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def thread_tokens(self, thread_id: str) -> int:
        thread = self._thread(thread_id)
        total = estimate_tokens(str(thread["summary"]))
        return total + sum(estimate_tokens(m["content"]) for m in thread["messages"])  # type: ignore[union-attr]

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})  # type: ignore[union-attr]
        if self.thread_tokens(thread_id) > self.threshold_tokens:
            self._compact(thread_id)

    def _compact(self, thread_id: str) -> None:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        if len(messages) <= self.keep_messages:
            return
        older, recent = messages[: -self.keep_messages], messages[-self.keep_messages :]
        previous = [{"role": "summary", "content": thread["summary"]}] if thread["summary"] else []
        thread["summary"] = summarize_messages(previous + older, max_items=self.summary_max_items)
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])
