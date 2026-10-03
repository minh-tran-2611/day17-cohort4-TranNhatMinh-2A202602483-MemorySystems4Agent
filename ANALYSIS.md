# Phân tích kết quả benchmark (Day 17)

Chạy: `python src/benchmark.py` (chế độ offline, deterministic) · `pytest src/test_agents.py -v` (9 test pass).
Cấu hình: compact threshold = 1000 tokens, giữ lại 4 message gần nhất, token ước lượng ≈ số ký tự / 4.

## Kết quả

### Standard Benchmark (`data/conversations.json`, 10 phiên)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |              1889 |                   13466 |                 0.00 |             0.20 |                     0 |           0 |
| Advanced |              2227 |                   29049 |                 1.00 |             1.00 |                   515 |           0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`, 16 lượt dài)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |              2408 |                   21828 |                 0.00 |             0.20 |                     0 |           0 |
| Advanced |              2450 |                   11709 |                 1.00 |             1.00 |                   321 |           3 |

## Ba lớp memory được tách riêng

| Lớp | Nơi triển khai | Phạm vi |
|---|---|---|
| Short-term | `SessionState.messages` (baseline), `CompactMemoryManager.state[thread]["messages"]` (advanced) | trong một `thread_id` |
| Persistent | `UserProfileStore` → `state/profiles/<user>/User.md` | qua mọi thread / session |
| Compact | `CompactMemoryManager` + `summarize_messages()` | nén phần cũ của một thread dài |

## 1. Vì sao Advanced có recall tốt hơn Baseline

- Câu hỏi recall được hỏi ở **thread mới**. Baseline bắt đầu thread mới với danh sách message rỗng nên chỉ trả lời "chưa có thông tin" → recall 0. Đây là hành vi đúng thiết kế, test `test_cross_session_recall` kiểm tra rằng baseline vẫn nhớ trong cùng thread nhưng quên khi sang thread mới.
- Advanced trích fact ổn định (tên, nơi ở, nghề, style, sở thích, đồ uống, món ăn, thú cưng) vào `User.md` ở mỗi lượt, rồi trả lời từ file đó → recall 1.0 trên cả hai bộ dữ liệu.
- Hai agent dùng chung hàm `answer_from_facts()`. Chúng chỉ khác nhau ở **nguồn** fact, nên đây là phép so sánh công bằng.

## 2. Vì sao Advanced tốn hơn ở hội thoại ngắn

- Mỗi phiên standard chỉ khoảng 20 message ngắn (khoảng 400 tokens), chưa bao giờ chạm ngưỡng 1000 tokens → **0 lần compact**. Advanced vẫn phải giữ toàn bộ lịch sử như baseline.
- Mỗi lượt Advanced còn kéo theo system prompt dài hơn và toàn bộ `User.md` (khoảng 130 tokens). Vì vậy `Prompt tokens processed` lớn gấp **2.16 lần** baseline (29049 so với 13466).
- `Agent tokens only` cao hơn khoảng 18% vì Advanced báo lại các fact đã cập nhật vào User.md và trả lời đầy đủ hơn.
- Kết luận: với hội thoại ngắn, persistent memory là **chi phí cố định cho mỗi lượt**. Chi phí này chỉ đáng bỏ ra khi người dùng quay lại nhiều phiên.

## 3. Vì sao compact giúp Advanced ở hội thoại dài

- Với baseline, prompt của lượt thứ *i* chứa cả *i* lượt trước đó → tổng prompt tăng theo **bậc hai** khi số lượt tăng.
- Advanced compact **3 lần**: chỉ giữ 4 message gần nhất, phần còn lại nén thành summary tối đa 6 bullet. Prompt mỗi lượt vì thế gần như **bị chặn trên**, bằng system + User.md + summary + 4 message.
- Kết quả: `Prompt tokens processed` giảm **46%** (11709 so với 21828), trong khi recall vẫn là 1.0, vì fact quan trọng đã nằm trong `User.md`, không phụ thuộc vào summary.
- `Agent tokens only` gần như bằng nhau (2450 so với 2408). Compact **không** làm giảm số token sinh ra hay số token người dùng gõ. Nó chủ yếu tối ưu **ngữ cảnh phải kéo theo** ở mỗi lượt, tức `prompt tokens processed`.
- Hội thoại càng dài thì khoảng cách càng lớn (test `test_compact_reduces_prompt_load_on_long_thread` yêu cầu giảm ít nhất 40%).

## 4. Tăng trưởng file memory và rủi ro

- `User.md` của `dungct` tăng lên 515 bytes sau 10 phiên, của `dungct_stress` là 321 bytes. File không lưu nguyên văn hội thoại, chỉ lưu fact.
- Các guardrail chống phình file:
  - Fact dạng scalar (nơi ở, nghề nghiệp) bị **ghi đè** khi có fact mới, không cộng dồn.
  - Fact dạng tập hợp (style, sở thích) bị cắt còn tối đa 8 mục.
  - Change log chỉ giữ 10 dòng gần nhất.
- Rủi ro:
  - **Lưu sai fact**: rule-based extraction có thể bắt nhầm câu đùa hoặc câu giả định. Một fact sai sẽ được đưa vào **mọi** prompt về sau, nên lỗi bị khuếch đại.
  - **Fact lỗi thời**: sở thích cũ không tự hết hạn (chưa làm memory decay).
  - **Quyền riêng tư**: `User.md` là dữ liệu cá nhân dạng plaintext, cần có chính sách xoá (`UserProfileStore.reset`) và không commit lên git (thư mục `state/` đã nằm trong `.gitignore`).
  - **Summary mất chi tiết**: summary heuristic chỉ giữ câu đầu của mỗi message cũ. Chi tiết tin tức trong stress test sẽ bị mất, chỉ còn fact trong User.md là chắc chắn được giữ.

## 5. Phần bonus đã làm

| Bonus | Cách làm | Lợi ích | Rủi ro / chi phí |
|---|---|---|---|
| Confidence threshold | `extract_profile_candidates()` gán độ tin cậy cho từng fact: câu có từ rào đón ("có lẽ", "đang cân nhắc", "dự định") được 0.4, câu đính chính ("đính chính", "giờ", "không còn") được 0.95. Chỉ ghi vào User.md khi độ tin cậy ≥ `PROFILE_CONFIDENCE_THRESHOLD` (mặc định 0.6) | Tránh ghi các kế hoạch chưa chắc chắn thành fact | Có thể bỏ sót fact thật nếu người dùng nói vòng vo |
| Conflict handling | `upsert_fact()` ghi đè giá trị cũ và chỉ giữ giá trị cũ trong Change log, ví dụ `location: Đà Nẵng -> Huế`, `profession: backend engineer -> MLOps engineer` | Không bao giờ có hai fact mâu thuẫn trong Profile. Recall đúng fact mới nhất (Huế, MLOps, Đà Nẵng trong stress test) | Nếu một correction bị trích sai, nó sẽ ghi đè lên fact đúng |
| Chống lưu câu hỏi / nhiễu | `is_question()` bỏ qua các câu hỏi và yêu cầu nhắc lại. Các từ báo nhiễu ("đùa", "chỉ là nơi ... họp", "đừng lấy ... làm nơi ở") và cụm phủ định ("không còn làm", "đừng nói") chặn việc trích fact | "product manager" và "Hà Nội" trong stress test không bị lưu | Rule cứng theo tiếng Việt, khó mở rộng sang các cách diễn đạt mới |
| Entity extraction có cấu trúc | User.md lưu theo các field cố định (`- key: value`), đọc lại bằng `facts()` | Prompt gọn, dễ diff, dễ test | Schema cố định: fact nằm ngoài schema sẽ bị bỏ qua |

Chưa làm: **memory decay**. Hướng làm: lưu thêm `last_seen` cho từng fact và giảm ưu tiên hoặc bỏ khỏi prompt các mục lâu không được nhắc lại.

## 6. Chế độ live (LLM thật)

Đặt các biến sau trong `.env`: `LAB_MODE=live`, `LLM_PROVIDER` (openai | custom | gemini | anthropic | ollama | openrouter), `LLM_MODEL` và API key tương ứng.

- Baseline dùng `create_agent` + `InMemorySaver`.
- Advanced dùng thêm:
  - tool `read_user_memory` / `update_user_fact`
  - `dynamic_prompt` để chèn User.md vào system prompt
  - `SummarizationMiddleware` cho thread dài
- Benchmark sẽ dùng `JUDGE_*` model để chấm `Response quality`.

Đã kiểm tra rằng agent live dựng được với cả 6 provider. **Chưa chạy với API thật** vì repo chưa có `.env` / API key.
