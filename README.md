# Nykaa Domain Support Agent (LangGraph)

Completed the **Nykaa (E-commerce & Retail)** track.
Everything runs under `MOCK_LLM` with zero API keys.

## How to Reproduce
Run the project scripts in sequence from the project root:

    python dataset.py
    python chunking.py
    python index_kb.py
    python check_index.py

Outputs match `transcripts/task1_dataset.txt`, `transcripts/task3_index.txt`, and `transcripts/task3_sample_retrieval.txt` deterministically.

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

## Knowledge Base & Dual-Collection Indexing (Tasks 2 & 3)
- **Knowledge Base (`kb/`)**: 12 plain-prose policy documents (each exactly 4 sentences, 408–559 characters) covering return windows, COD refunds, delivery SLAs, reverse pickup, brand warranty, order cancellation, loyalty points, payment failures, size exchange, damaged items, international shipping, and escalation workflows.
- **KB Refinement & Freeze Note**: After the first retrieval check, `return_window.txt` was reworded to name `makeup, cosmetics, skincare` explicitly because the Beauty sentence originally only said `Beauty` (causing MiniLM to rank the Electronics sentence higher on cosmetic item queries). A query-specific version containing `'lipstick'` was tried and rejected as overfitting. The KB is frozen after this edit.
- **Chunking Strategies (`chunking.py`)**:
  - **Fixed-Size (`kb_fixed`)**: Sliding window with `chunk_size=200` and `overlap=40` (guarded by `assert overlap < chunk_size`). Produces **40 chunks** across the 12 files (average length: **176.8 characters**).
  - **Sentence-Based (`kb_sentence`)**: Regex boundary splitter `(?<=[.!?])\s+`. Produces **48 chunks** across the 12 files—4 standalone sentences per document (average length: **123.4 characters**).
- **Vector Store (`index_kb.py`)**:
  - Embedded locally using `sentence-transformers/all-MiniLM-L6-v2`.
  - Indexed into persistent ChromaDB (`chroma_db/`) across two collections (`kb_fixed` and `kb_sentence`) with `metadata={"hnsw:space": "cosine"}`.
  - Idempotent rebuilds: existing collections are cleanly deleted and recreated on each run.
  - Chunk IDs follow `<doc_id>::<strategy>::<idx>` with `{"doc_id", "chunk_index", "strategy"}` metadata for automated Precision@3 and Recall@3 calculation.
- **Sample Retrieval Check (`check_index.py`)**:
  - Query: `"How long do I have to return a lipstick?"`
  - Similarity computed as `1.0 - cosine_distance`.
  - `kb_fixed` Rank 1: `return_window` (Chunk 0, Sim `0.5188`), Rank 2: `return_window` (Chunk 1, Sim `0.4551`), Rank 3: `damaged_item` (Chunk 2, Sim `0.4096`).
  - `kb_sentence` Rank 1: `return_window` (Chunk 1 — exact Beauty 5-day return policy sentence, Sim `0.5182`), Rank 2: `return_window` (Chunk 2, Sim `0.4635`), Rank 3: `return_window` (Chunk 0, Sim `0.4472`).
