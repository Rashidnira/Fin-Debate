# Fin-Debate: A Multi-Agent Deliberation Framework for Automated Fact-checking in Financial Domain

This repo contains the code and dataset for our paper on evidence-grounded financial fact-checking using adversarial multi-agent deliberation.

The core idea: instead of relying on a single LLM pass (which tends to hedge toward "half-true"), we use multiple agents with opposing perspectives — a supportive analyst, a contradictory analyst, and a critical analyst — then let a judge synthesize their disagreements into a final verdict. Combined with an iterative evidence retrieval pipeline that validates temporal alignment, this approach achieves ~63% macro F1 on a challenging 3-class financial claim dataset, outperforming baselines by about 10 points.

## Architecture Overview

The system has two main modules:

**Evidence Preparation Module** (`evidence_collection.py`)
- Evidence Extractor — pulls relevant facts from documents, checks temporal alignment from google retrived top 15 results again each claim
- Evidence Verifier — decides if the extracted evdience is sufficient to make verdict
- Query Generator — creates targeted Google queries to fill gaps produced by evdinece verifier
- Conflict Resolver — adds verified supplementary info when evidence is still incomplete

These 4 agents work in a loop: extract from local KB → verify → if insufficient, generate queries → search the web → extract from results → verify again. Up to 2 iterations.

**Decision-Making Module** (`fact_check_verdicts.py`)
- Supportive Analyst — argues in favor of the claim
- Contradictory Analyst — argues against the claim
- Critical Analyst — weighs both sides
- Judge — reads all three analyses, makes the final call

The three analysts run independently (no access to each other's output) to prevent anchoring bias. The judge is explicitly instructed not to default to HALF-TRUE as a safe option.

## Dataset

`dataset_Fin_misinformation.csv` — 300 financial claims from PolitiFact (2007–2025), balanced across three classes:

| Label | Count | Description |
|-------|-------|-------------|
| TRUE | 100 | Accurate, nothing significant missing |
| HALF-TRUE | 100 | Partially accurate, missing context |
| FALSE | 100 | Not accurate |

We started with 2,168 claims scraped using financial keywords and S&P 500 company names, then collapsed the original 5 PolitiFact labels into 3 (mostly-true → true, mostly-false → false) and sampled a balanced subset. Claims cover corporate earnings, tax policy, fiscal policy, and market-level assertions from 183 unique speakers.

## Results

| Method | Macro P (%) | Macro R (%) | Macro F1 (%) |
|--------|-------------|-------------|--------------|
| Zero-Shot | 53.4 | 52.3 | 52.7 |
| RAG-Dense (FAISS) | 53.4 | 52.7 | 52.8 |
| RAG-BM25 | 53.0 | 52.3 | 52.4 |
| Delphi Agent | 62.0 | 53.3 | 53.6 |
| **FIN-Debate (ours)** | **63.8** | **64.0** | **63.4** |

The biggest gains come from adversarial deliberation rather than better retrieval — removing it drops F1 by 8 points (ablation study in the paper).

## Setup

### Requirements

```
pip install pandas langchain-openai pydantic serpapi aiohttp beautifulsoup4 scikit-learn tabulate
```

### API Keys

You'll need:
- `OPENAI_API_KEY` — for GPT-4o-mini calls
- `SERPAPI_API_KEY` — for Google search during evidence collection

```bash
export OPENAI_API_KEY="your-key-here"
export SERPAPI_API_KEY="your-key-here"
```

## How to Run

### Step 1: Build local knowledge base

Before running evidence collection, you need a local KB for each claim. We used SerpAPI to fetch the top 15 Google results per claim and saved the full text under `data/kb/claim_{id}/fulltext/`. This step is separate from the main pipeline.

### Step 2: Collect evidence

```bash
python evidence_collection.py \
    --dataset data/dataset_Fin_misinformation.csv \
    --kb-path data/kb/ \
    --output-dir outputs/evidence
```

This produces `final_evidence_complete.jsonl` — one JSON object per claim with all collected evidence, verification results, and query logs.

Use `--sample 10` to test on just the first 10 claims.

### Step 3: Generate verdicts

```bash
python fact_check_verdicts.py \
    --evidence outputs/evidence/final_evidence_complete.jsonl \
    --dataset data/dataset_Fin_misinformation.csv \
    --output-dir outputs/verdicts
```

This runs all 4 decision-making agents and produces `fact_check_verdicts.jsonl` with predictions and full agent outputs.

### Step 4: Compute metrics

```bash
python compute_metrics.py \
    --input data/all_verdicts_300.csv \
    --output-dir results/
```

Prints overall and per-class metrics, confusion matrices, and exports `metrics_summary.csv`.

## Repo Structure

```
.
├── README.md
├── evidence_collection.py      # Evidence Preparation Module (4 agents)
├── fact_check_verdicts.py      # Decision-Making Module (4 agents)
├── compute_metrics.py          # Evaluation metrics computation
├── dataset_Fin_misinformation.csv  # 300 balanced financial claims
└── outputs/                    # generated at runtime
    ├── evidence/
    │   └── final_evidence_complete.jsonl
    └── verdicts/
        └── fact_check_verdicts.jsonl
```

## Key Design Decisions

- **Temporal filtering**: Evidence must fall within ±2 years of the claim date. We found this critical for financial claims where numbers change year to year — without it, the model often validates claims using outdated or future statistics.

- **Fact-check site exclusion**: We block PolitiFact, Snopes, and FactCheck.org from evidence retrieval to prevent data leakage from ground-truth sources.

- **Adversarial over consensus**: Unlike Delphi which uses consensus-driven deliberation, our agents argue from fixed opposing stances. This produces richer analytical signals and reduces the tendency to hedge toward HALF-TRUE.

- **GPT-4o-mini throughout**: All agents use the same model to keep the comparison fair across experiments. The performance differences come from architecture, not model capacity.



## Citation

Paper citation will be added upon publication.
