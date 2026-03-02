#!/usr/bin/env python3
"""
Iterative evidence collection pipeline for financial fact-checking.
Uses 4 LLM agents to gather, verify, and supplement evidence for claims.

Run: python evidence_collection.py --dataset data/claims.csv --kb-path data/kb/
"""

import os, re, json, asyncio, argparse, logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from serpapi import GoogleSearch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- config ---
MODEL = "gpt-4o-mini"
MAX_ITER = 2
TEMPORAL_TOL = 2  # years
MAX_KB_DOCS = 12
SCRAPE_TIMEOUT = 15
MAX_CONTENT = 15_000

# sites we want to exclude to avoid data leakage
FACTCHECK_SITES = ["politifact.com", "factcheck.org", "snopes.com"]
SOCIAL = ["x.com", "twitter.com", "reddit.com", "youtube.com",
          "facebook.com", "instagram.com"]

_sem = asyncio.Semaphore(3)


# --- pydantic models for structured llm output ---

class EvidenceExtraction(BaseModel):
    have_evidence: bool = Field(description="True if evidence is sufficient and credible")
    evidence: str = Field(description="Extracted relevant evidence with context")
    source: str = Field(description="Source document name or URL")
    source_type: str = Field(default="KB")
    temporal_alignment: Optional[str] = Field(default=None)
    evidence_year: Optional[int] = Field(default=None)

class EvidenceVerification(BaseModel):
    verdict: str = Field(description="sufficient | insufficient")
    missing_elements: List[str] = Field(description="Missing claim elements")
    reason: str

class QueryGeneration(BaseModel):
    working_claim: str
    subclaims: List[str] = Field(default_factory=list)
    missing_elements: List[str] = Field(description="Gaps in evidence")
    queries: List[str] = Field(description="1-2 search queries")
    reason: str

class SupplementaryAddition(BaseModel):
    info: str
    source: str
    source_link: str
    addresses_gap: str

class ConflictSolverOutput(BaseModel):
    have_verified_info: bool
    additions: List[SupplementaryAddition] = Field(default_factory=list)
    reasoning: str


# ---- date/temporal helpers ----

def parse_year(date_str):
    """Try to pull a 4-digit year out of whatever date format we got."""
    if not date_str or date_str == "Unknown":
        return None
    m = re.search(r"(19|20)\d{2}", str(date_str))
    if m:
        return int(m.group())
    # try common formats as fallback
    for fmt in ("%d-%b-%y", "%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(str(date_str), fmt).year
        except ValueError:
            continue
    return None

def _yr_range(year):
    if year is None:
        return None, None
    return year - TEMPORAL_TOL, year + TEMPORAL_TOL

def check_temporal(text, claim_year):
    """Check if evidence text mentions years close enough to claim year."""
    if not claim_year:
        return True, None, "UNKNOWN"
    years_found = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")]
    if not years_found:
        return True, None, "UNKNOWN"

    lo, hi = _yr_range(claim_year)
    good_years = [y for y in years_found if lo <= y <= hi]
    if good_years:
        best = min(good_years, key=lambda y: abs(y - claim_year))
        return True, best, "GOOD"

    closest = min(years_found, key=lambda y: abs(y - claim_year))
    if abs(closest - claim_year) <= 5:
        return True, closest, "MODERATE"
    return False, closest, "POOR"

def temporal_filter(ev_list, claim_year):
    """Drop evidence that's too far from the claim year."""
    if not claim_year:
        return ev_list, 0, {"status": "no_claim_year"}
    kept, n_removed = [], 0
    for ev in ev_list:
        ok, best_yr, alignment = check_temporal(ev.get("evidence", ""), claim_year)
        if ok:
            ev["temporal_alignment"] = alignment
            ev["evidence_year"] = best_yr
            kept.append(ev)
        else:
            n_removed += 1
    return kept, n_removed, {"claim_year": claim_year, "kept": len(kept), "removed": n_removed}


# ---- data loading ----

def load_dataset(path):
    df = pd.read_csv(path)
    # some datasets have space in column name
    if "source post" in df.columns:
        df.rename(columns={"source post": "source_post"}, inplace=True)
    return df

def load_kb(claim_id, kb_base):
    """Load local KB text files for a given claim."""
    p = Path(kb_base) / f"claim_{claim_id}" / "fulltext"
    if not p.exists():
        return []
    return [Document(page_content=f.read_text(encoding="utf-8"),
                     metadata={"source": f.name}) for f in p.glob("*.txt")]

def filter_kb_docs(docs):
    """Remove factcheck sites and social media from KB docs."""
    filtered = []
    for doc in docs:
        txt = doc.page_content.lower()
        # skip if it's from politifact etc
        if re.search(r"URL\s*:\s*.*politifact\.com", doc.page_content, re.IGNORECASE):
            continue
        if any(s in txt for s in FACTCHECK_SITES + SOCIAL):
            continue
        filtered.append(doc)
    return filtered[:MAX_KB_DOCS]


# ---- llm wrapper ----

async def call_llm(system, user, schema, variables):
    """Simple structured LLM call with concurrency limit."""
    async with _sem:
        llm = ChatOpenAI(model=MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_messages([("system", system), ("user", user)])
        chain = prompt | llm.with_structured_output(schema)
        return await chain.ainvoke(variables)


# =====================================================
# Agent 1: Evidence Extractor
# =====================================================
async def run_extractor(claim, speaker, date, docs, src_type="KB"):
    yr = parse_year(date)
    lo, hi = _yr_range(yr)

    sys_prompt = f"""You are an Evidence Extractor tasked with identifying and extracting content from the documents that is relevant to the claim.
Follow these steps:
1. Identify Key Points in the Claim: Extract the main ideas or key points from the original claim.
2. Analyze document Content: Locate sections containing information relevant to the claim.
3. Extract Relevant Content: Extract sentences relevant to the claim - INCLUDING BOTH SUPPORTING AND CONTRADICTING EVIDENCE.
4. Ensure Clarity: Resolve ambiguous pronouns by replacing them with clear nouns.
5. Look for temporal relevance of the claim and evidence.
6. Extract Source Information: The document contains "URL:" field - extract this URL as the source.
7. Determine sufficiency: Set have_evidence TRUE if evidence contains specific facts directly addressing the claim, source is credible and timeframe matches. Set FALSE if only tangential mentions, wrong time period, or unreliable sources.
8. Do not use social media posts as credible evidence.

CRITICAL - INCLUDE DOCUMENT DATE: Always include publication date or year in extracted evidence.
TEMPORAL RULE: Claim is from {yr}. Only extract evidence dated within {lo}-{hi}.
CIRCULAR REFERENCE RULE: Do NOT extract if document only quotes {speaker} making the same claim.
SOURCE FILTER RULE: Set have_evidence FALSE if source URL contains politifact.com, factcheck.org, or snopes.com.

Return JSON only for: have_evidence, evidence, source."""

    usr_prompt = """Speaker: {source_post}  Date: {date}  Claim: {claim}
News Report:
{report}
Return JSON only."""

    results = []
    for doc in docs:
        try:
            content = doc.page_content
            if "URL:" not in content:
                content = f"URL: {doc.metadata.get('source', 'Unknown')}\n\n{content}"
            res = await call_llm(sys_prompt, usr_prompt, EvidenceExtraction,
                                 {"source_post": speaker, "date": date,
                                  "claim": claim, "report": content})
            if res.have_evidence:
                res.source_type = src_type
                results.append(res)
        except Exception as e:
            log.warning("extractor error: %s", e)
    return results


# =====================================================
# Agent 2: Evidence Verifier
# =====================================================
async def run_verifier(claim, speaker, date, evidence_list):
    yr = parse_year(date)
    lo, hi = _yr_range(yr)

    sys_prompt = f"""You are an Evidence Verifier. Determine if evidence is SUFFICIENT to verify the claim.

SUFFICIENCY CRITERIA:
- "sufficient" if claim can be verified/refuted with credible, specific, temporally aligned evidence.
- "insufficient" if key elements missing, evidence vague, time-misaligned, or unreliable.

TEMPORAL ALIGNMENT: Claim is from {yr}. Evidence MUST be from {lo}-{hi}.
If evidence uses "currently"/"today" WITHOUT explicit year from that range, mark "insufficient" and add "temporally ambiguous evidence" to missing_elements.
TEMPORAL CUES: If claim says "last year" and date is {date}, convert to absolute year.

Return JSON: verdict, missing_elements, reason."""

    usr_prompt = """Speaker: {source_post}  Date: {date}  Claim: {claim}
Evidence:
{evidence}
Return JSON only."""

    combined = "\n---\n".join(
        f"Source: {e.source}\n{e.evidence}" for e in evidence_list
    ) or "No evidence."

    try:
        return await call_llm(sys_prompt, usr_prompt, EvidenceVerification,
                              {"source_post": speaker, "date": date,
                               "claim": claim, "evidence": combined})
    except Exception as e:
        return EvidenceVerification(
            verdict="insufficient", missing_elements=["verification failed"], reason=str(e))


# =====================================================
# Agent 3: Query Generator
# =====================================================
async def run_query_gen(claim, verification, speaker, date):
    yr = parse_year(date)

    sys_prompt = f"""You are a Query Generator creating Google search queries to find missing evidence.
Rules:
- 5-10 words per query, include specific entities/numbers/dates.
- ALWAYS include {yr} in queries for temporal relevance.
- Use source_post to resolve ALL pronouns with actual names.
- Only query for MISSING elements. Return at most 2 queries.

Return JSON: working_claim, subclaims, missing_elements, queries, reason."""

    # double braces for the variables that langchain will fill in,
    # single brace for the f-string year
    usr_prompt = """Claim: {{claim}}  Source: {{source_post}}  Date: {{date}} (Year: {yr})
Verdict: {{verdict}}  Missing: {{missing_elements}}  Reason: {{reason}}
Return JSON only.""".format(yr=yr)

    try:
        res = await call_llm(sys_prompt, usr_prompt, QueryGeneration,
                             {"claim": claim, "source_post": speaker, "date": date,
                              "verdict": verification.verdict,
                              "missing_elements": ", ".join(verification.missing_elements),
                              "reason": verification.reason})
        # cap at 2 queries and make sure year is in there
        res.queries = (res.queries or [])[:2]
        if yr:
            res.queries = [q if str(yr) in q else f"{q} {yr}" for q in res.queries]
        return res
    except Exception:
        fallback_q = f"{claim[:50]} {yr}" if yr else claim
        return QueryGeneration(working_claim=claim, queries=[fallback_q],
                               reason="Fallback", missing_elements=[])


# =====================================================
# Agent 4: Conflict Resolver
# =====================================================
async def run_conflict_resolver(claim, speaker, date, evidence_list, gaps):
    yr = parse_year(date)

    sys_prompt = f"""You are a Conflict Solver. Provide VERIFIED supplementary information to fill evidence gaps.

RULES:
1. ONLY provide information you are CERTAIN about from widely-known, verified facts.
2. If NOT SURE, set have_verified_info = FALSE.
3. NEVER make up facts, statistics, or sources.
4. ALWAYS provide a real, verifiable source with a link.
5. Better to say "beyond my knowledge" than provide uncertain information.

Claim is from {yr}. Info should be relevant to that time period.

Return JSON: have_verified_info, additions (list with info/source/source_link/addresses_gap), reasoning."""

    usr_prompt = """Speaker: {source_post}  Date: {date}  Claim: {claim}
Gaps: {missing_elements}
Current Evidence:
{evidence}
Return JSON only."""

    ev_text = "\n".join(
        f"{i}. [{e.get('source_type')}] {e.get('evidence', '')}"
        for i, e in enumerate(evidence_list, 1)
    ) or "None."
    gap_text = "\n".join(f"- {g}" for g in gaps) if gaps else "None"

    try:
        return await call_llm(sys_prompt, usr_prompt, ConflictSolverOutput,
                              {"source_post": speaker, "date": date, "claim": claim,
                               "missing_elements": gap_text, "evidence": ev_text})
    except Exception as e:
        return ConflictSolverOutput(have_verified_info=False, additions=[],
                                    reasoning=f"Error: {e}")


# ---- web search + scraping ----

async def search_web(query):
    """Run google search via serpapi."""
    def _do_search():
        raw = GoogleSearch({
            "q": query, "api_key": os.environ["SERPAPI_API_KEY"], "num": 8
        }).get_dict().get("organic_results", [])
        return [{"url": r.get("link", ""), "title": r.get("title", ""),
                 "snippet": r.get("snippet", "")} for r in raw]
    try:
        return await asyncio.get_running_loop().run_in_executor(None, _do_search)
    except Exception:
        return []

def filter_results(results):
    """Remove factcheck sites and social media from search results."""
    bad = FACTCHECK_SITES + ["twitter.com", "facebook.com", "youtube.com", "x.com"]
    out = [r for r in results[:10] if not any(d in r.get("url", "").lower() for d in bad)]
    return out[:6]

async def _scrape_single(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=SCRAPE_TIMEOUT),
                               ssl=False, headers={"User-Agent": "Mozilla/5.0"}) as resp:
            if resp.status != 200:
                return None
            ctype = resp.headers.get("content-type", "").lower()
            if "pdf" in ctype:
                return None
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            for junk in soup(["script", "style", "nav", "header", "footer"]):
                junk.decompose()
            body = soup.find("main") or soup.find("article") or soup
            text = body.get_text("\n", strip=True)
            text = "\n".join(ln.strip() for ln in text.split("\n") if ln.strip())
            return text[:MAX_CONTENT] if len(text) > 100 else None
    except Exception:
        return None

async def scrape_all(urls):
    sem = asyncio.Semaphore(5)
    async def _go(sess, item):
        async with sem:
            content = await _scrape_single(sess, item["url"])
            return {**item, "content": content}
    conn = aiohttp.TCPConnector(limit=5, ssl=False)
    async with aiohttp.ClientSession(connector=conn) as sess:
        tasks = [_go(sess, u) for u in urls]
        done = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in done if isinstance(r, dict) and r.get("content")]

async def extract_web_evidence(claim, speaker, date, scraped_pages):
    """Run extractor on scraped web pages with tighter temporal focus."""
    yr = parse_year(date)
    lo, hi = _yr_range(yr)

    sys_prompt = f"""You are an Evidence Extractor. Find evidence specifically from {yr} (+/-2 years: {lo}-{hi}).

The web page may contain data from MULTIPLE years. You MUST:
1. Search the page for data from {lo}-{hi}
2. Ignore data outside this range
3. ALWAYS include the year in extracted evidence
4. Set have_evidence TRUE only if data found from that range

CIRCULAR REFERENCE RULE: Do NOT extract if document only quotes {speaker} making the claim.
SOURCE FILTER RULE: Set have_evidence FALSE if URL contains politifact.com, factcheck.org, snopes.com.

Return JSON: have_evidence, evidence, source."""

    usr_prompt = """Claim: {claim}  Speaker: {source_post}  Date: {date}  Target: {claim_year} ({year_min}-{year_max})
URL: {url}
Content:
{content}
Return JSON only."""

    out = []
    for page in scraped_pages:
        try:
            res = await call_llm(
                sys_prompt, usr_prompt, EvidenceExtraction,
                {"source_post": speaker, "date": date, "claim": claim,
                 "claim_year": yr, "year_min": lo, "year_max": hi,
                 "url": page["url"], "content": page["content"]})
            if res.have_evidence:
                res.source = page["url"]
                res.source_type = "WEB"
                out.append(res)
        except Exception:
            pass  # just skip failed pages
    return out


# =====================================================
# Main processing loop for a single claim
# =====================================================

def _evidence_to_dicts(ev_objects):
    """Convert pydantic evidence objects to plain dicts with source_link."""
    dicts = [e.model_dump() for e in ev_objects]
    for d in dicts:
        d["source_link"] = d["source"]
    return dicts

async def process_claim(df, claim_id, kb_base):
    row = df[df["claim_id"] == claim_id]
    if row.empty:
        return None
    row = row.iloc[0]

    claim = row["claim"]
    speaker = row.get("source_post", "Unknown")
    date = row.get("date", "Unknown")
    yr = parse_year(date)

    log.info("Claim %d | year=%s | %s", claim_id, yr, claim[:80])

    all_ev = []
    t_stats, queries_log, all_gaps = [], [], []
    iter_log = []
    last_verification = None

    # -- KB extraction --
    docs = filter_kb_docs(load_kb(claim_id, kb_base))
    if docs:
        kb_results = await run_extractor(claim, speaker, date, docs, "KB")
        kb_dicts = _evidence_to_dicts(kb_results)
        if yr:
            kb_dicts, _, stats = temporal_filter(kb_dicts, yr)
            t_stats.append({"stage": "KB", **stats})
        all_ev.extend(kb_dicts)

    # -- iterative loop: verify -> generate queries -> web search -> extract --
    for it in range(MAX_ITER):
        # convert back to pydantic for verifier
        ev_objs = [EvidenceExtraction(**e) if isinstance(e, dict) else e for e in all_ev]
        verif = await run_verifier(claim, speaker, date, ev_objs)
        last_verification = verif
        all_gaps.extend(verif.missing_elements)

        n_good = sum(1 for e in all_ev if e.get("temporal_alignment") == "GOOD")
        n_unk = sum(1 for e in all_ev if e.get("temporal_alignment") == "UNKNOWN")

        # if everything is temporally unknown, don't trust the "sufficient" label
        if n_good == 0 and n_unk > 0 and verif.verdict == "sufficient":
            verif.verdict = "insufficient"
            verif.missing_elements.append(f"Need explicit year from {yr}")

        iter_log.append({
            "iteration": it + 1, "evidence_count": len(all_ev),
            "verdict": verif.verdict, "missing": verif.missing_elements,
            "temporal_good": n_good, "temporal_unknown": n_unk
        })

        if (verif.verdict == "sufficient" and n_good > 0) or it == MAX_ITER - 1:
            break

        # need more evidence - generate queries and search
        qgen = await run_query_gen(claim, verif, speaker, date)
        queries_log.append({
            "iteration": it + 1, "queries": qgen.queries,
            "missing_elements": verif.missing_elements, "reason": qgen.reason
        })

        # run web searches
        raw_results = []
        for q in qgen.queries:
            raw_results.extend(filter_results(await search_web(q)))
        # deduplicate
        seen_urls = set()
        unique = [r for r in raw_results if r["url"] not in seen_urls and not seen_urls.add(r["url"])]

        pages = await scrape_all(unique)
        if not pages:
            continue

        web_results = await extract_web_evidence(claim, speaker, date, pages)
        web_dicts = _evidence_to_dicts(web_results)
        if yr and web_dicts:
            web_dicts, _, stats = temporal_filter(web_dicts, yr)
            t_stats.append({"stage": f"WEB_iter{it+1}", **stats})
        all_ev.extend(web_dicts)

    # -- conflict resolver --
    unique_gaps = list(set(all_gaps))
    solver = await run_conflict_resolver(claim, speaker, date, all_ev, unique_gaps)
    if solver.have_verified_info:
        for a in solver.additions:
            all_ev.append({
                "have_evidence": True, "evidence": a.info,
                "source": a.source, "source_link": a.source_link,
                "source_type": "SUPPLEMENTARY",
                "temporal_alignment": "UNKNOWN", "evidence_year": yr
            })

    # count by source type
    def _cnt(t): return sum(1 for e in all_ev if e.get("source_type") == t)

    return {
        "claim_id": claim_id, "claim_text": claim, "source_post": speaker,
        "date": date, "claim_year": yr,
        "total_evidence": len(all_ev),
        "kb_evidence_count": _cnt("KB"),
        "web_evidence_count": _cnt("WEB"),
        "supplementary_evidence_count": _cnt("SUPPLEMENTARY"),
        "evidence": all_ev,
        "verification_result": {
            "verdict": last_verification.verdict if last_verification else "unknown",
            "missing_elements": last_verification.missing_elements if last_verification else [],
            "reason": last_verification.reason if last_verification else ""
        },
        "queries_used": queries_log,
        "supplementary_info": {
            "have_verified_info": solver.have_verified_info,
            "additions": [a.model_dump() for a in solver.additions],
            "reasoning": solver.reasoning
        },
        "iteration_log": iter_log,
        "temporal_filtering_stats": t_stats
    }


# =====================================================
# CLI
# =====================================================
async def main():
    ap = argparse.ArgumentParser(description="Evidence collection for financial fact-checking")
    ap.add_argument("--dataset", required=True, help="Path to claims CSV")
    ap.add_argument("--kb-path", required=True, help="Knowledge base directory")
    ap.add_argument("--output-dir", default="outputs/evidence")
    ap.add_argument("--sample", type=int, default=None, help="Only process first N claims")
    args = ap.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY"
    assert os.environ.get("SERPAPI_API_KEY"), "Set SERPAPI_API_KEY"
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, "final_evidence_complete.jsonl")

    df = load_dataset(args.dataset)
    ids = df["claim_id"].tolist()[:args.sample]
    log.info("Processing %d claims -> %s", len(ids), outfile)

    results = []
    for i, cid in enumerate(ids, 1):
        log.info("[%d/%d] claim %d", i, len(ids), cid)
        try:
            r = await process_claim(df, cid, args.kb_path)
            if r:
                results.append(r)
                # save after each claim so we don't lose progress
                with open(outfile, "w") as f:
                    for item in results:
                        f.write(json.dumps(item) + "\n")
        except Exception as e:
            log.error("claim %d failed: %s", cid, e)

    n_suf = sum(1 for r in results if r["verification_result"]["verdict"] == "sufficient")
    log.info("Done: %d/%d processed, %d had sufficient evidence", len(results), len(ids), n_suf)


if __name__ == "__main__":
    asyncio.run(main())
