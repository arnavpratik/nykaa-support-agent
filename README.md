# Nykaa Domain Support Agent (LangGraph)

Completed the **Nykaa (E-commerce & Retail)** track.
Everything runs under `MOCK_LLM` with zero API keys.

## How to Reproduce
Run the dataset generator in one command from the project root:

    python dataset.py

Output matches `transcripts/task1_dataset.txt` deterministically.

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