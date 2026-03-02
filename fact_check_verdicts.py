#!/usr/bin/env python3
"""
Multi-Agent Fact-Checking Verdict Generator — FIN-Debate
=========================================================
Four agents with adversarial deliberation: Supportive, Contradictory,
and Critical analysts independently evaluate claims, then a Judge
synthesizes their analyses into a final verdict.

Input:  evidence JSONL (from evidence_collection.py) + ground-truth CSV
Output: verdict JSONL with per-claim predictions

Usage:
    python fact_check_verdicts.py --evidence data/evidence.jsonl --dataset data/claims.csv
    python fact_check_verdicts.py --evidence data/evidence.jsonl --dataset data/claims.csv --sample 10

Environment: OPENAI_API_KEY
"""

import os, json, asyncio, argparse, logging
from typing import List

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL = "gpt-4o-mini"
VERDICT_LABELS = {"TRUE", "FALSE", "HALF-TRUE"}
_semaphore = asyncio.Semaphore(3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_evidence(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def load_ground_truth(path: str) -> dict:
    df = pd.read_csv(path)
    return {int(r["claim_id"]): str(r.get("ground_truth", "UNKNOWN")).upper().strip()
            for _, r in df.iterrows()}

def format_evidence(evidence_list: List[dict]) -> str:
    if not evidence_list:
        return "No evidence available."
    return "\n\n".join(
        f"Evidence {i} [{ev.get('source_type', 'Unknown')}]: {ev.get('evidence', '')}\nSource: {ev.get('source', 'Unknown')}"
        for i, ev in enumerate(evidence_list, 1))

def normalize_verdict(v: str) -> str:
    return v.upper().strip().replace(" ", "-")

def parse_final_verdict(text: str) -> str:
    if "FINAL VERDICT:" not in text:
        return "UNKNOWN"
    line = text.split("FINAL VERDICT:")[1].split("\n")[0].upper()
    if "HALF" in line:
        return "HALF-TRUE"
    if "TRUE" in line:
        return "TRUE"
    if "FALSE" in line:
        return "FALSE"
    return "UNKNOWN"

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
async def llm_call(system: str, user: str, variables: dict) -> str:
    async with _semaphore:
        llm = ChatOpenAI(model=MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_messages([("system", system), ("user", user)])
        result = await (prompt | llm).ainvoke(variables)
        return result.content

# ---------------------------------------------------------------------------
# Shared verdict definitions used by all analyst agents
# ---------------------------------------------------------------------------
VERDICT_DEFS = """POLITIFACT VERDICT DEFINITIONS:
- TRUE: The statement is accurate and there's nothing significant missing.
- HALF-TRUE: The statement is partially accurate but leaves out important details or takes things out of context.
- FALSE: The statement is not accurate."""

ANALYST_USER = """CLAIM: {claim}
SPEAKER: {source}
YEAR: {year}

EVIDENCE:
{evidence}"""

# ===================================================================
# AGENT 1 — Supportive Analyst
# ===================================================================
async def analyze_support(claim: str, source: str, year: int, evidence: str) -> str:
    system = f"""You are a fact-checker. Analyze evidence from a SUPPORTIVE perspective.
Find evidence that SUPPORTS the claim.

{VERDICT_DEFS}

Output format:
VERDICT: [TRUE/FALSE/HALF-TRUE]
EXPLANATION: [explanation citing evidence]
TEMPORAL ALIGNMENT: [GOOD/MODERATE/POOR]
SOURCES:
1. [Source] - [URL]"""
    return await llm_call(system, ANALYST_USER, {"claim": claim, "source": source, "year": year, "evidence": evidence})

# ===================================================================
# AGENT 2 — Contradictory Analyst
# ===================================================================
async def analyze_contradict(claim: str, source: str, year: int, evidence: str) -> str:
    system = f"""You are a fact-checker. Analyze evidence from a CONTRADICTING perspective.
Find evidence that CONTRADICTS the claim. Look for missing context or overstatements.

{VERDICT_DEFS}

Output format:
VERDICT: [TRUE/FALSE/HALF-TRUE]
EXPLANATION: [explanation citing evidence]
TEMPORAL ALIGNMENT: [GOOD/MODERATE/POOR]
SOURCES:
1. [Source] - [URL]"""
    return await llm_call(system, ANALYST_USER, {"claim": claim, "source": source, "year": year, "evidence": evidence})

# ===================================================================
# AGENT 3 — Critical Analyst
# ===================================================================
async def analyze_critical(claim: str, source: str, year: int, evidence: str) -> str:
    system = f"""You are a fact-checker. Provide CRITICAL, balanced analysis.
Weigh both supporting and contradicting evidence.

{VERDICT_DEFS}

Output format:
VERDICT: [TRUE/FALSE/HALF-TRUE]
EXPLANATION: [balanced explanation]
TEMPORAL ALIGNMENT: [GOOD/MODERATE/POOR]
SOURCES:
1. [Source] - [URL]"""
    return await llm_call(system, ANALYST_USER, {"claim": claim, "source": source, "year": year, "evidence": evidence})

# ===================================================================
# AGENT 4 — Judge
# ===================================================================
async def judge_verdict(claim: str, source: str, year: int,
                        support: str, contradict: str, critical: str) -> tuple:
    system = f"""You are the JUDGE. Review all 3 agents and give FINAL VERDICT.

{VERDICT_DEFS}

IMPORTANT - BE STRICT AND DECISIVE:
- Do NOT default to HALF-TRUE as a safe option.
- HALF-TRUE is only for claims that are genuinely partially accurate with missing context.
- If evidence clearly supports the claim → TRUE
- If evidence clearly contradicts the claim → FALSE
- Make a firm decision based on the weight of evidence, not to play it safe.

CITATION REQUIREMENTS:
- Cite sources by NAME inline (e.g., "According to New York Times...").
- List all source links at the end.

Format:
=== FINAL VERDICT: [TRUE/FALSE/HALF-TRUE] ===

EXPLANATION:
[Reasoning with inline source citations]

SOURCES:
- [source link 1]
- [source link 2]"""

    user = """CLAIM: {claim}
SPEAKER: {source}
YEAR: {year}

=== SUPPORT ===
{support}

=== CONTRADICT ===
{contradict}

=== CRITICAL ===
{critical}"""

    result = await llm_call(system, user,
                            {"claim": claim, "source": source, "year": year,
                             "support": support, "contradict": contradict, "critical": critical})
    return result, parse_final_verdict(result)

# ===================================================================
# Pipeline: process one claim
# ===================================================================
async def process_claim(claim_data: dict, ground_truth_dict: dict) -> dict:
    claim_id = claim_data["claim_id"]
    claim_text = claim_data["claim_text"]
    source_post = claim_data.get("source_post", "Unknown")
    year = claim_data.get("claim_year", 0)
    evidence_text = format_evidence(claim_data.get("evidence", []))
    ground_truth = ground_truth_dict.get(claim_id, "UNKNOWN")

    support = await analyze_support(claim_text, source_post, year, evidence_text)
    contradict = await analyze_contradict(claim_text, source_post, year, evidence_text)
    critical = await analyze_critical(claim_text, source_post, year, evidence_text)
    judge, final_verdict = await judge_verdict(claim_text, source_post, year, support, contradict, critical)

    correct = normalize_verdict(final_verdict) == normalize_verdict(ground_truth)
    log.info("  Claim %d: %s (GT: %s) %s", claim_id, final_verdict, ground_truth, "✓" if correct else "✗")

    return {
        "claim_id": claim_id, "claim": claim_text, "source": source_post,
        "year": year, "ground_truth": ground_truth,
        "support": support, "contradict": contradict, "critical": critical,
        "judge": judge, "final_verdict": final_verdict, "prediction_correct": correct,
    }

# ===================================================================
# Entry point
# ===================================================================
async def main():
    parser = argparse.ArgumentParser(description="Multi-agent fact-checking verdict generator")
    parser.add_argument("--evidence", required=True, help="Evidence JSONL from evidence_collection.py")
    parser.add_argument("--dataset", required=True, help="Ground-truth claims CSV")
    parser.add_argument("--output-dir", default="outputs/verdicts", help="Output directory")
    parser.add_argument("--sample", type=int, default=None, help="Process first N claims")
    args = parser.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY"
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "fact_check_verdicts.jsonl")

    evidence_list = load_evidence(args.evidence)[:args.sample]
    gt = load_ground_truth(args.dataset)
    log.info("Processing %d claims → %s", len(evidence_list), output_file)

    results, correct = [], 0
    for i, claim_data in enumerate(evidence_list, 1):
        log.info("[%d/%d] Claim %d", i, len(evidence_list), claim_data["claim_id"])
        try:
            result = await process_claim(claim_data, gt)
            results.append(result)
            if result["prediction_correct"]:
                correct += 1
            with open(output_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
        except Exception as e:
            log.error("Claim %d failed: %s", claim_data["claim_id"], e)

    log.info("Done: %d/%d correct (%.1f%%)", correct, len(results), correct / len(results) * 100 if results else 0)

if __name__ == "__main__":
    asyncio.run(main())
