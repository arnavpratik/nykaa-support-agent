# Nykaa Domain Support Agent (LangGraph)

Completed the **Nykaa (E-commerce & Retail)** track.
Everything runs under `MOCK_LLM` with zero API keys and zero network access (after the one-time embedding-model download).

## How to Reproduce
Run the project scripts in sequence from the project root:

    # Day 1: Dataset Generation
    python dataset.py

    # Day 2: KB Chunking, Vector Indexing & Verification
    python chunking.py
    python index_kb.py
    python check_index.py

    # Day 3: Calibration, Grounded Generation, Evaluation & Tool Execution
    python calibrate.py
    python run_grounded.py
    python evaluate_chunking.py
    python run_tool.py

    # Day 4: State Machine, Memory, Schema & Guardrail Transcripts
    python run_agent.py t7
    python run_agent.py t8_multi
    python run_agent.py t8_fresh
    python run_agent.py t9
    python run_agent.py t10

    # Day 5: FastAPI Service, Logging Audit & RAG Triad Evaluation
    python run_api.py
    python eval_triad.py

    # Day 6: MCP, SQLite checkpointing, resilience
    # MCP needs two terminals (server blocks; port 8000 must be free, so stop uvicorn first)
    #   Terminal 1: python mcp_server.py
    #   Terminal 2: python mcp_client.py
    python run_checkpoint.py stage1
    python run_checkpoint.py stage2
    python run_resilience.py retry
    python run_resilience.py node_timeout
    python run_resilience.py global_timeout

    # Interactive FastAPI Server (Swagger UI at http://127.0.0.1:8000/docs)
    uvicorn api:app --reload

Outputs match files in `transcripts/` deterministically.

---

## Dataset Design Choices (Task 1)
- **Seed**: `42` (Landed within the target band on the first run; no re-seeding or weight adjustments were needed).
- **Total Records**: `60`
- **Field Draw Sequence**: Each record draws `category`, `status`, `order_value_inr`, `days_since_created`, then the `delayed_shipment` roll in strict sequential order from a single isolated `random.Random(42)` instance.
- **Category Weights**: Apparel `0.25`, Electronics `0.15`, Home `0.15`, Footwear `0.15`, Beauty `0.30` (Matches `CATEGORIES` declaration order).
- **Status Weights**: Placed `0.15`, Shipped `0.25`, Delivered `0.40`, Returned `0.10`, Refunded `0.10` (Matches `STATUSES` declaration order).
- **Price Range Reasoning**: Ranges reflect authentic Nykaa basket sizes, where cosmetic items and home decor are lower-to-mid ticket while personal styling appliances and tech gadgets command higher ticket values.
- **Price Ranges (INR)**:
  - Apparel: ₹700 – ₹6,000
  - Electronics: ₹1,500 – ₹12,000
  - Home: ₹500 – ₹4,000
  - Footwear: ₹900 – ₹5,000
  - Beauty: ₹200 – ₹3,500
- **delayed_shipment Rule & Probabilities**:
  - *Definition*: Missed promised dispatch or delivery SLA date. Flag is always `False` for `Returned` and `Refunded` orders (a deliberate simplification: the flag only describes orders not yet returned or refunded).
  - *Probabilities*:
    - `Returned` / `Refunded`: `0.0`
    - `Placed` / `Shipped`: `0.45` if `days_since_created >= 5`, else `0.05`
    - `Delivered`: `0.20` if `days_since_created >= 8`, else `0.0`
- **Validation Results**:
  - Categories: Beauty (20), Apparel (15), Footwear (9), Home (9), Electronics (7) [All >= 3: PASS]
  - Statuses: Delivered (20), Shipped (17), Placed (10), Refunded (7), Returned (6) [All >= 1: PASS]
  - Delayed Shipment Rate: **20.00%** (12/60) [Within 10.0%–30.0%: PASS]
  - Age Range: [0, 30] verified for all records
  - ID Uniqueness: 60 unique non-numeric IDs (`ORD-0001` to `ORD-0060`)

---

## Knowledge Base & Dual-Collection Indexing (Tasks 2 & 3)
- **Knowledge Base (`kb/`)**: 12 plain-prose policy documents (each exactly 4 sentences, 408–559 characters) covering return windows, COD refunds, delivery SLAs, reverse pickup, brand warranty, order cancellation, loyalty points, payment failures, size exchange, damaged items, international shipping, and escalation workflows.
- **KB Refinement & Freeze Note**: After the first retrieval check, `return_window.txt` was reworded to name `makeup, cosmetics, skincare` explicitly because the Beauty sentence originally only said `Beauty`. A query-specific version containing `'lipstick'` was tried and rejected as overfitting. The KB is permanently frozen after this edit.
- **Chunking Strategies (`chunking.py`)**:
  - **Fixed-Size (`kb_fixed`)**: Sliding window with `chunk_size=200` and `overlap=40` (guarded by `assert overlap < chunk_size`). Produces **40 chunks** across the 12 files (average length: **176.8 characters**).
  - **Sentence-Based (`kb_sentence`)**: Regex boundary splitter `(?<=[.!?])\s+`. Produces **48 chunks** across the 12 files—4 standalone sentences per document (average length: **123.4 characters**).
- **Vector Store (`index_kb.py`)**:
  - Embedded locally using `sentence-transformers/all-MiniLM-L6-v2`.
  - Indexed into persistent ChromaDB (`chroma_db/`) across two collections (`kb_fixed` and `kb_sentence`) with `metadata={"hnsw:space": "cosine"}`.
  - Chunk IDs follow `<doc_id>::<strategy>::<idx>` with `{"doc_id", "chunk_index", "strategy"}` metadata for automated evaluation.
- **Sample Retrieval Check (`check_index.py`)**:
  - Query: `"How long do I have to return a lipstick?"`
  - `kb_fixed` Rank 1: `return_window` (Sim `0.5188`), Rank 2: `return_window` (Sim `0.4551`), Rank 3: `damaged_item` (Sim `0.4096`).
  - `kb_sentence` Rank 1: `return_window` (Sim `0.5182`), Rank 2: `return_window` (Sim `0.4635`), Rank 3: `return_window` (Sim `0.4472`).

---

## Retrieval Calibration & Grounded Generation (Task 4)
- **Retrieval Engine (`rag_core.py`)**: Singleton loader for `SentenceTransformer` and `ChromaDB` client. Converts cosine distance to similarity via `1.0 - distance`. Configured with `MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"`.
- **Threshold Calibration (`calibrate.py`)**:
  - Evaluated top-1 similarity across 6 diverse in-scope customer queries and 5 out-of-scope queries on both collections.
  - For the deployed `kb_sentence` collection, purely unrelated queries cluster tightly in `[0.0779, 0.1236]` (`[0.0848, 0.1510]` on `kb_fixed`), while in-scope queries score in `[0.3453, 0.7956]`.
  - A near-miss corporate query (*"What is Nykaa's current share price on the stock exchange?"*) scored `0.4771` on `kb_sentence` because financial tokens overlap `loyalty_points`. Raising the similarity threshold to `0.49` would improperly reject valid indirect customer questions like *"My shoes don't fit"* (`0.3453`).
  - **Operational Decision**: Retrieval confidence threshold for `kb_sentence` is calibrated at **`0.30`** (between max unrelated `0.1236` and min in-scope `0.3453`), paired with a deterministic corporate/non-support keyword guardrail to catch corporate near-misses.
- **Grounded Generation (`generate.py` & `run_grounded.py`)**:
  - Queries below `0.30` similarity or matching corporate keywords return: `"I don't know based on the Nykaa knowledge base."` with empty sources (`grounded=False`).
  - Demonstrated on 5 in-scope queries plus 2 out-of-scope queries: a pure off-topic query ("How do I bake a chocolate cake?", similarity `0.0779`, rejected by threshold) and a corporate near-miss ("What is Nykaa's current share price on the stock exchange?", similarity `0.4771`, rejected by the out-of-scope keyword filter).

---

## Chunking Evaluation & Recommendation (Task 5)
Evaluated using 5 distinct in-scope queries against pre-established ground truth parent documents (`evaluate_chunking.py`). Retrieved chunks are deduplicated to parent `doc_id`s prior to computing metrics (precision denominator = count of unique retrieved documents):

| Strategy | Mean Precision@3 | Mean Recall@3 | Mean Top-1 Similarity |
|---|---|---|---|
| **`kb_fixed`** | **0.7000** | **1.0000** | **0.4785** |
| **`kb_sentence`** | **0.7000** | **1.0000** | **0.5158** |

- **Tie-Breaking Recommendation**:
  Both strategies **tied** on document-level Mean Precision@3 (`0.7000`) and Mean Recall@3 (`1.0000`). We break the tie using the secondary signal of **Mean Top-1 Similarity (`0.5158` for `kb_sentence` vs. `0.4785` for `kb_fixed`)** and select `kb_sentence` for deployment. Intact policy sentences avoid mid-clause cuts and improve rank-1 semantic matching on specific policy inquiries (e.g., COD refund scores `0.7956` vs. `0.6773`).

---

## Order Status & Escalation Scoring Tool (Task 6)
- **Tool Implementation (`tools.py` & `run_tool.py`)**:
  - Provides `check_order_status(record_id: str) -> dict` with case/whitespace normalization (`"  ord-0008  "` resolves cleanly) and structured error dicts for unknown IDs.
- **Escalation Formula & Resolved-Order Rule**:
  $$\text{recency} = \min\left(\frac{\text{days\_since\_created}}{30}, 1.0\right)$$
  $$\text{delay\_term} = 1.0 \text{ if delayed\_shipment else } 0.0$$
  - **Open orders (`Placed`, `Shipped` — 27 of 60 records)**:
    $$\text{score} = 0.6 \times \text{delay\_term} + 0.4 \times \text{recency}$$
  - **Resolved orders (`Delivered`, `Returned`, `Refunded` — 33 of 60 records)**:
    $$\text{score} = 0.0$$
  - *Justification*: Delivered packages have reached the customer, so they require no active dispatch SLA escalation. This aligns with `kb/escalation_matrix.txt`, which restricts automated escalation to unresolved SLA breaches.
- **Zero-Inflated Distribution & Percentile Threshold**:
  - Because 33 of 60 orders are resolved (`score = 0.0`), the 60-order distribution is zero-inflated:
    - `P50`: **0.0000** | `P75`: **0.3000** | `P80`: **0.3733** | `P90`: **0.6960**
  - **Decision Rule**: Threshold is set to **`0.3733`** (the **80th percentile** of all 60 orders), effectively selecting the top urgency tier among open orders. Exactly **13 of 60 orders** exceed this threshold.

---

## LangGraph Architecture, Schema, Memory & Guardrails (Tasks 7–10)

### 1. State Machine Topology (`agent.py`)
State contains: `query`, `conversation_id`, `trace_id`, `masked_query`, `history`, `last_order_id`, `extracted_order_id`, `route`, `guardrails_triggered`, `result`, `response`.

- **Nodes (7 total)**:
  1. `node_load_memory`: Loads conversational turn history and `last_order_id` from disk.
  2. `node_input_guard`: Redacts PII, checks for prompt injection, and flags out-of-scope queries.
  3. `node_classify_intent`: Evaluates masked text for `ORD-\d{4}` regex or order keywords with remembered `last_order_id`; routes to `order_status` or defaults to `rag`.
  4. `node_order_tool`: Calls `check_order_status` and formats structured order response.
  5. `node_rag_tool`: Executes grounded retrieval via `answer(collection_name="kb_sentence")`. When `[PHONE_MASKED]` or `[CARD_LAST4_MASKED]` is present in `masked_query`, filters out the PII filler clause before embedding so placeholder tokens do not dilute semantic similarity below `0.30`.
  6. `node_output_guard`: Validates groundedness under `MOCK_LLM` (checks provenance against retrieved text).
  7. `node_finalize`: Enforces JSON Schema validation, persists turn to disk, and emits final payload.

- **Conditional Edges (2 total)**:
  - **Edge 1 (`route_after_input_guard`)**: If `input_guard` flags prompt injection or out-of-scope, routes directly to `finalize` with `route="refused"`, bypassing tools. Otherwise routes to `classify_intent`.
  - **Edge 2 (`route_after_classify`)**: Routes to `order_tool` if `route == "order_status"`, else routes to `rag_tool`.

### 2. Guardrails & Limitations (`guardrails.py`)
- **PII Redaction (`mask_pii`)**:
  - Phone: Indian 10-digit mobile numbers with optional `+91`/`91`, spaces, or hyphens: `(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)`.
  - Card Last-4: Tightened context-anchored regex `(?i)((?:card\s+)?(?:ending(?:\s+(?:in|with))?|last\s*(?:4|four)(?:\s*digits)?)[^\d]{0,10})(\d{4})(?!\d)`.
  - *Limitation & False Positive Resistance*: Explicit context anchors prevent false positives on years and dates (e.g., `"my card was delivered in 2025"` leaves `2025` untouched; `"pincode 110001"` and `"ORD-0012"` remain untouched). A bare 4-digit number mentioned without card anchor phrases would not be masked.
- **Prompt Injection Detection (`detect_injection`)**:
  - Evaluates 8 heuristic patterns: `ignore (all )?(previous|prior) instructions`, `disregard .*(rules|instructions)`, `reveal .*system prompt`, `you are now`, `act as\b`, `developer mode`, `jailbreak`, `pretend (you are|to be)`.
  - *Limitation*: Regex heuristic catching known phrasing patterns; novel, obfuscated, or multi-turn conversational jailbreaks require an external LLM safety classifier.
- **Corporate Out-of-Scope Filter (`is_out_of_scope`)**:
  - Matches 10 corporate keywords: `share price`, `stock price`, `stock exchange`, `market cap`, `investor`, `ipo`, `salary`, `careers`, `founder`, `ceo`.
  - *Limitation*: Keyword-based filter rather than semantic classifier; non-standard corporate inquiries not containing these tokens fall back to retrieval confidence thresholding.
- **Output Groundedness (`check_grounded`)**:
  - Validates `grounded=True`, non-empty `sources`, `top_similarity >= 0.30`, and under `MOCK_LLM`, verifies that `top_chunk_text` is present verbatim in the output answer.

### 3. Response Schema (`schema.py`)
Draft-07 JSON Schema enforcing `additionalProperties: false` with required fields:
- `trace_id` (string), `route` (enum: `rag`, `order_status`, `refused`), `answer` (string), `sources` (array of strings), `grounded` (boolean), `guardrails_triggered` (array of strings), `data` (object or null).

### 4. Conversation Memory (`memory.py`)
- **File**: `memory_store.json` shaped as `{conversation_id: {"messages": [{"user": masked_query, "agent": answer, "route": route}], "last_order_id": str | None}}`.
- **Atomic Persistence**: Uses `tempfile.NamedTemporaryFile` + `os.replace` to prevent corrupted partial reads.
- **Disk Privacy**: Stored messages contain strictly sanitized `masked_query` strings. Raw PII is never serialized to disk.

---

## FastAPI Endpoints & Request Auditing (Tasks 11 & 12)

### 1. Endpoint Interface
| Method | Endpoint | Request Body | Response Body | Description |
|---|---|---|---|---|
| `GET` | `/health` | None | `{"status": "ok"}` | Liveness check |
| `POST` | `/ask` | `AskRequest`: `query: str [1..1000]`, `conversation_id: Optional[str]` | `AskResponse`: `trace_id`, `route`, `answer`, `sources`, `grounded`, `guardrails_triggered`, `data` | End-to-end customer support agent execution |
| `POST` | `/add-document` | `AddDocumentRequest`: `doc_id: str ^[a-z0-9_]+$`, `text: str [>=10]` | `AddDocumentResponse`: `doc_id`, `chunks_added`, `collection` | Dynamic policy ingestion (syncs `kb_sentence` & `kb_fixed`) |

- **Duplicate Prevention**: `/add-document` returns HTTP `409 Conflict` if `doc_id` already exists in ChromaDB metadata.
- **Frozen Baseline Note**: Dynamically added documents are verified and immediately cleaned up (`delete(where={"doc_id": ...})`) during automated testing to keep the frozen evaluation index intact at 40 fixed and 48 sentence chunks.

### 2. Forensic JSONL Request Auditing (`logging_utils.py`)
Every incoming request (including validation failures like `422` and conflicts like `409`) logs one record to `logs/requests.jsonl`:
- **Fields**: `trace_id`, `timestamp` (UTC ISO), `endpoint`, `conversation_id`, `request_text` (masked), `route`, `guardrails_triggered`, `grounded`, `status_code`, `duration_ms`, `answer_length`, `error_message`.
- **Mandatory Masking Guarantee**: `mask_pii()` runs *internally* inside `log_request()`. Even if an endpoint handler or exception passes unscrubbed user text, raw phone numbers and credit card numbers can never reach the disk audit log. Verified via `Select-String -Pattern "9876543210|4321"` with zero matches.

---

## RAG Triad Evaluation (Task 13)

### 1. Methodology & Metric Definitions
Evaluated across 15 distinct test queries: 12 in-scope policy queries (1 per KB topic) + 3 edge cases (corporate near-miss, pure off-topic, ambiguous return). Order status tracking queries route to the deterministic tool and are excluded from RAG triad metrics.

1. **Context Relevance**: Arithmetic mean of the cosine similarities of the top-3 retrieved chunks from `retrieve(query, collection_name="kb_sentence", k=3)`. Calculated directly without filtering by the operational `0.30` threshold to maintain metric independence.
2. **Groundedness**: Under extractive `MOCK_LLM`, measures the proportion of answer content sentences present in the retrieved chunks.
   - For answered queries: `1.0` if the extracted sentence appears in retrieved context (near `1.00` by construction under extractive mock).
   - For out-of-scope refusals / fallbacks (Queries #13 & #14): `1.0` by definition since no ungrounded claims were asserted.
   - Safeguard rule: For an in-scope query (`expected_doc != None`) where the agent refuses/falls back, score `0.0` to penalize false refusals. (Note: No in-scope query refused in the final evaluation run, so this safeguard did not affect the reported averages).
3. **Answer Relevance**: Cosine similarity between query embedding and answer embedding.
   - For correct refusals on out-of-scope queries (#13 share price and #14 off-topic cake), a correct refusal scores `1.0000` by definition.
   - For an in-scope false refusal, score `0.0000` (user received no answer).

### 2. Evaluation Results Summary
| ID | Topic / Intent | Route | Context Rel. | Groundedness | Answer Rel. | Top Source / Behavior |
|---|---|---|---|---|---|---|
| 1 | Return Window | `rag` | 0.4590 | 1.00 | 0.5191 | `return_window` |
| 2 | COD Refund | `rag` | 0.5262 | 1.00 | 0.5749 | `cod_refund` |
| 3 | Delivery SLAs | `rag` | 0.4788 | 1.00 | 0.5288 | `delivery_sla` |
| 4 | Reverse Pickup | `rag` | 0.4135 | 1.00 | 0.4149 | `reverse_pickup` |
| 5 | Warranty | `rag` | 0.2850 | 1.00 | 0.2913 | `return_window` (Wrong-doc answer) |
| 6 | Order Cancellation | `rag` | 0.5024 | 1.00 | 0.5318 | `order_cancellation` |
| 7 | Loyalty Points | `rag` | 0.5362 | 1.00 | 0.5671 | `loyalty_points` |
| 8 | Payment Failure | `rag` | 0.4735 | 1.00 | 0.5572 | `payment_failure` |
| 9 | Size Exchange | `rag` | 0.2847 | 1.00 | 0.4318 | `size_exchange` |
| 10 | Damaged Item | `rag` | 0.4018 | 1.00 | 0.4531 | `damaged_item` |
| 11 | International Shipping | `rag` | 0.4112 | 1.00 | 0.4515 | `international_shipping` |
| 12 | Escalation Matrix | `rag` | 0.5197 | 1.00 | 0.4962 | `escalation_matrix` |
| 13 | Corporate Near-Miss | `refused` | 0.4177 | 1.00 | 1.0000 | Guardrail refusal |
| 14 | Pure Off-Topic | `rag` | 0.0881 | 1.00 | 1.0000 | Ungrounded fallback |
| 15 | Ambiguous Return | `rag` | 0.3668 | 1.00 | 0.4380 | `return_window` |

- **Sum & Average Verification**:
  - **12 In-Scope Sums**: C.Rel `5.2920` | Groundedness `12.0000` | Answer Rel `5.8177`
  - **12 In-Scope Averages**: Context Relevance **`0.4410`** | Groundedness **`1.0000`** | Answer Relevance **`0.4848`**
    *(Note: Answer Relevance `0.4848` reflects inclusion of Query #5 at `0.2913` due to wrong-document retrieval).*
  - **3 Edge Cases Sum**: C.Rel `0.8726` | Groundedness `3.0000` | Answer Rel `2.4380` (#13: `1.0` + #14: `1.0` + #15: `0.4380`)
  - **All 15 Evaluated Sums**: C.Rel `6.1646` | Groundedness `15.0000` | Answer Rel `8.2557`
  - **All 15 Evaluated Averages**: Context Relevance **`0.4110`** | Groundedness **`1.0000`** | Answer Relevance **`0.5504`**

### 3. Critical Analytical Findings & Limitations
1. **Query #5 as a Wrong-Document Failure**:
   - For Query #5 (*"My hair straightener stopped working after two weeks, who handles repairs?"*), the expected document was `warranty`.
   - The agent retrieved `return_window` as its top chunk because *"hair straightener"* matched the Electronics sentence (*"Electronics and styling appliances must be returned within 7 days..."*) with top-1 similarity exceeding `0.30`.
   - **Groundedness is blind to this failure**: Groundedness scored `1.00` because the generated text faithfully matched the retrieved `return_window` chunk verbatim.
   - **The triad works as a system**: Context Relevance (`0.2850`) and Answer Relevance (`0.2913`) cleanly expose the defect—the agent cited a 7-day return window instead of answering who handles manufacturer warranty repairs.
2. **Extractive Mock Groundedness**: Under `MOCK_LLM`, groundedness is `1.00` by construction for answered queries. It does not validate real-world abstractive summarization, but confirms that ungrounded queries (#14) fall back without hallucination.
3. **Ambiguity Over-Retrieval**: Query #15 (*"Can I return it?"*) retrieved `return_window` (`0.3668`) despite zero item context, demonstrating that under-specified queries can pass retrieval barriers when tokens overlap general policy titles.

---

## MCP, SQLite Checkpointing & Resilience (Tasks 14–16)

### Task 14: MCP server and client
- `mcp_server.py` (FastMCP) exposes one tool, `check_order_status`, at `http://127.0.0.1:8000/mcp`. It imports the same function from `tools.py`, so the agent and the MCP server share one implementation. The tool docstring covers the `ORD-0012` ID format, the returned fields (`status`, `order_value_inr`, `escalation_score`, `escalate`, `reason`) and the unknown-ID behavior.
- `mcp_client.py` is a separate process. It lists the tools and calls `check_order_status` for three IDs:
  - `ORD-0001`: Placed, delayed, score `0.6933`, escalate `True`
  - `ORD-0008`: Returned, score `0.0`
  - `ORD-9999`: structured error dict (order not found)
- Run: Terminal 1 `python mcp_server.py`, Terminal 2 `python mcp_client.py`.

### Task 15: SQLite checkpointing and resume
- `SqliteSaver` with `checkpoints.sqlite`, thread ID `ckpt-demo-01`, compiled with `interrupt_before=["rag_tool"]`.
- **Stage 1** (`python run_checkpoint.py stage1`): runs `load_memory`, `input_guard` and `classify_intent`, then stops before `rag_tool`. The process exits.
- **Stage 2** (`python run_checkpoint.py stage2`, a new process): loads the saved state for the same thread, resumes, and runs only `rag_tool`, `output_guard` and `finalize`.
- Each node prints an `[EXEC]` marker when it actually runs. The stage 2 transcript shows no markers for the first three nodes, which proves they were loaded from the checkpoint and not re-executed.
- Eight checkpoint rows are printed directly from the SQLite file.
- Scope: the API graph has no checkpointer. This graph is for the demo only.

### Task 16: Timeouts and retries
All nodes are async, so `asyncio.wait_for` genuinely cancels the slow call. A timeout on a sync node cannot stop a blocking thread.

| Mechanism | Configuration | Result |
|---|---|---|
| Retry (LangGraph `RetryPolicy`) | `max_attempts=4`, `initial_interval=0.2s`, `backoff_factor=2.0`, `max_interval=2.0s`, `jitter=True`, `retry_on=TransientError` | Node fails twice, succeeds on attempt 3 (1.428s total; jitter adds up to 1s per retry). Exhaustion after 4 attempts raises cleanly. |
| Per-node timeout | `@with_timeout(seconds=1.0)` raising `NodeTimeoutError`, not in `retry_on` | Fired at 1.005s on a 5s sleep |
| Global timeout | `asyncio.wait_for(graph.ainvoke(...), timeout=2.0)` | Fired at 2.013s. Node 3 was cancelled and node 4 never started (four 0.8s nodes). |

Intervals are short so the demo runs quickly. Production values would be larger.

---

## Transcript-Per-Task Verification Map (Tasks 1–16)
All 23 verification artifacts in `transcripts/` are non-empty and deterministically reproducible:

| Task | Description | Transcript File(s) |
|---|---|---|
| **Task 1** | Synthetic Order Dataset (`seed=42`, 60 orders, 20.00% delayed) | `transcripts/task1_dataset.txt` |
| **Task 2** | 12-Document Nykaa Policy Knowledge Base | `kb/*.txt` (verified in `task3_index.txt`) |
| **Task 3** | Dual Chunking (`kb_fixed`: 40, `kb_sentence`: 48) & Sample Retrieval | `transcripts/task3_index.txt`<br>`transcripts/task3_sample_retrieval.txt` |
| **Task 4** | Threshold Calibration (`0.30`) & Grounded Generation | `transcripts/task4_calibration.txt`<br>`transcripts/task4_grounded.txt` |
| **Task 5** | Quantitative Chunking Evaluation (`P@3=0.7000`, `R@3=1.0000`) | `transcripts/task5_eval.txt` |
| **Task 6** | Order Status Tool & 80th-Percentile Escalation (`0.3733`) | `transcripts/task6_tool.txt` |
| **Task 7** | LangGraph Conditional Routing (`rag` vs `order_status`) | `transcripts/task7_routing.txt` |
| **Task 8** | Conversational Memory (Multi-Turn Follow-up vs Fresh Session) | `transcripts/task8_memory_multiturn.txt`<br>`transcripts/task8_memory_fresh.txt` |
| **Task 9** | Output JSON Schema Validation (Valid + Invalid Rejection) | `transcripts/task9_schema.txt` |
| **Task 10** | Input/Output Guardrails (PII Masking, Injection, OOS, Groundedness) | `transcripts/task10_guardrails.txt` |
| **Task 11** | FastAPI Service (`GET /health`, `POST /ask`, `POST /add-document`) | `transcripts/task11_api.txt` |
| **Task 12** | PII-Scrubbed Forensic JSONL Audit Logging | `transcripts/task12_log_sample.jsonl` |
| **Task 13** | 15-Query RAG Triad Evaluation (`A.Rel=0.4848` in-scope / `0.5504` all-15) | `transcripts/task13_triad.txt`<br>`transcripts/task13_scores.json` |
| **Task 14** | MCP Server & Client Discovery and Execution | `transcripts/task14_mcp_server.txt`<br>`transcripts/task14_mcp_client.txt` |
| **Task 15** | Two-Stage Cross-Process SQLite Checkpointing (`SqliteSaver`) | `transcripts/task15_checkpoint_stage1.txt`<br>`transcripts/task15_checkpoint_stage2.txt` |
| **Task 16** | Resilience Wrappers (Retry with Jitter, Node & Global Timeouts) | `transcripts/task16_retry.txt`<br>`transcripts/task16_node_timeout.txt`<br>`transcripts/task16_global_timeout.txt` |

---

## Known Limitations

- **Heuristic guardrails and router:** the injection patterns, out-of-scope keywords and intent router catch known phrasings only. A follow-up such as "When will it arrive?" after an order lookup routes to RAG, because it has no tracking keyword.
- **PII scope:** only phone numbers and context-anchored card last-4 digits are masked. Names and addresses are not.
- **Mock groundedness:** `1.00` by construction, because the extractive mock copies retrieved text. Wrong-document retrieval (query #5) is visible only in context relevance and answer relevance.
- **Escalation distribution:** zero-inflated (33 of 60 orders are resolved and score 0), so the 80th percentile is effectively a percentile of open orders.
- **`sources` semantics:** lists documents retrieved above the threshold, not only the document quoted in the answer.
- **Checkpointed graph:** demo-only. The API graph uses `memory_store.json`.
