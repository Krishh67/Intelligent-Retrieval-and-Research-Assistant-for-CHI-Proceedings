"""
query_planner.py

Intelligent Retrieval Query Planner for CHI Research Assistant RAG System.

Performs ONE Gemini call per user query to simultaneously:
  1. Expand the user query into multiple semantically diverse sub-queries
     (Multi-Query Retrieval)
  2. Detect any implicit metadata constraints (year)
     (Metadata-Aware Retrieval)

The output JSON drives the full retrieval orchestration in the RAG pipeline.

Design Constraints:
  - ONE Gemini API call only (no polling, no chaining).
  - Gemini decides the number of queries (2–6) based on complexity.
  - Original user query is always preserved in the generated list.
  - Returns a validated Python dataclass so callers never touch raw JSON.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field-name aliases  (guards against occasional Gemini key-name typos)
# ---------------------------------------------------------------------------
# Maps any wrong key Gemini might emit → the correct required key.
# Normalization is applied immediately after json.loads, before validation.
KEY_ALIASES: dict = {
    "needs_year_year_filter":  "needs_year_filter",   # double-word typo
    "year_filter":             "needs_year_filter",   # shorthand
    "require_year_filter":     "needs_year_filter",   # synonym
    "requires_year_filter":    "needs_year_filter",   # synonym variant
    "filter_by_year":          "needs_year_filter",   # alternative phrasing
    "year_required":           "needs_year_filter",   # alternative phrasing
}

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryPlan:
    """
    Structured output from the query-planning Gemini call.

    Attributes:
        topic           : Concise topic label extracted by Gemini.
        year            : Detected publication year, or None.
        needs_year_filter: True when year filtering should be applied.
        queries         : Ordered list of retrieval queries (original first).
    """
    topic: str
    year: Optional[int]
    needs_year_filter: bool
    queries: List[str]


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are a retrieval query planner for an academic RAG system that indexes CHI (ACM Conference on Human Factors in Computing Systems) research papers.

Your only job is to analyse a user question and return a single JSON object (no markdown fences, no explanation, no extra keys).

JSON Schema you MUST follow:

{
"topic": <string — concise label for the core research topic>,
"year": <integer | null — detected publication year, null if not mentioned>,
"needs_year_filter": <boolean — true ONLY when the user explicitly requests a specific year>,
"queries": <array of strings — retrieval queries, between 1 and 6 items>
}

Rules for "queries":

1. The original user question MUST be the FIRST element.

2. Generate between 1 and 6 total queries.
   Generate additional queries ONLY when they meaningfully improve retrieval coverage.

3. Do NOT create simple paraphrases.
   Instead, cover different retrieval angles, subtopics, terminology, applications, methods, or concepts related to the topic.

4. Each additional query should retrieve papers that may not be found by the original wording.

5. Prefer concise retrieval-friendly phrases over conversational questions.

6. Avoid redundant queries that express the same intent using different wording.

7. Keep queries focused on semantic retrieval.

8. Never generate more than 6 queries ,generate 1-2 only,and only if retrieval will improve then generate more based on complexity.

Examples:

Bad:

* User trust in AI systems
* Trust development in AI systems
* Building trust in AI systems

Good:

* trust calibration in AI
* human-AI decision making
* reliance on AI recommendations
* explainable AI trust
* confidence in automation

Rules for "year":

* Set to an integer ONLY if the user explicitly names a year (e.g. "2024 CHI papers").
* Set to null for implicit or historical references (e.g. "recent papers").
* Valid years in the dataset: 2021, 2023, 2024.

Rules for "needs_year_filter":

* true → the user wants results restricted to a specific year.
* false → no year restriction needed.

Return valid JSON only. No markdown. No explanation.

CRITICAL — Key names must match the schema exactly.
Do NOT rename keys. Do NOT invent new keys.
The ONLY valid keys are: topic, year, needs_year_filter, queries.

"""

_PLANNER_USER_TEMPLATE = "User question: {question}"

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def analyze_and_expand_query(
    question: str,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> QueryPlan:
    """
    Analyse a user question and produce a QueryPlan via ONE Gemini call.

    This function combines multi-query expansion and metadata detection in a
    single LLM call, keeping API cost minimal.

    Args:
        question  : Raw user question string.
        model_name: Gemini model identifier (default: gemini-2.5-flash).
        api_key   : Optional API key override; falls back to the already-
                    configured genai client if omitted.

    Returns:
        QueryPlan dataclass with topic, year, filter flag, queries.

    Raises:
        ValueError : If Gemini returns malformed JSON after retries.
        RuntimeError: If the Gemini API call itself fails.

    Workflow:
        1. Build a deterministic prompt (system + user turn).
        2. Call Gemini with temperature=0.0 for reproducible JSON output.
        3. Strip any accidental markdown fences from the response.
        4. Parse JSON → validate required keys → build QueryPlan.
        5. Ensure the original question is always in queries[0].
    """
    logger.info("=" * 60)
    logger.info("QUERY PLANNER — Analyzing user question")
    logger.info(f"  Question: {question!r}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Optionally (re)configure genai if a key was explicitly passed
    # ------------------------------------------------------------------
    if api_key:
        genai.configure(api_key=api_key)

    # ------------------------------------------------------------------
    # 2. Build and send the planning prompt
    # ------------------------------------------------------------------
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_PLANNER_SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,           # deterministic JSON
                max_output_tokens=512,     # JSON is compact; 512 is plenty
            ),
        )
        user_message = _PLANNER_USER_TEMPLATE.format(question=question)
        response = model.generate_content(user_message)
        raw_text = response.text.strip()
        logger.info("✓ Received planning response from Gemini")
    except Exception as exc:
        raise RuntimeError(
            f"Gemini API call failed in query planner: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 3. Strip accidental markdown code fences if present
    # ------------------------------------------------------------------
    raw_text = _strip_markdown_fences(raw_text)

    # ------------------------------------------------------------------
    # 4. Parse and validate JSON
    # ------------------------------------------------------------------
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Query planner received non-JSON response from Gemini.\n"
            f"Raw response:\n{raw_text}\nJSON error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 4a. Normalize any mis-spelled field names Gemini occasionally emits
    # ------------------------------------------------------------------
    for wrong_key, correct_key in KEY_ALIASES.items():
        if wrong_key in data and correct_key not in data:
            logger.warning(
                f"  ⚠ Gemini used alias key '{wrong_key}' → correcting to '{correct_key}'"
            )
            data[correct_key] = data.pop(wrong_key)

    plan = _build_query_plan(question, data)

    # ------------------------------------------------------------------
    # 5. Emit structured logs
    # ------------------------------------------------------------------
    _log_query_plan(plan)

    return plan


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers that Gemini may add
    despite being asked not to.
    """
    # Pattern: optional language tag after opening fence
    pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(pattern, text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return text


def _build_query_plan(original_question: str, data: dict) -> QueryPlan:
    """
    Validate the parsed JSON dict and construct a QueryPlan.

    Applies safety defaults for any missing or mis-typed fields.
    """
    required_keys = {"topic", "year", "needs_year_filter", "queries"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(
            f"Query planner JSON is missing required keys: {missing}\n"
            f"Full data: {data}"
        )

    # --- topic ---
    topic = str(data["topic"]).strip() or "unknown topic"

    # --- year ---
    raw_year = data["year"]
    year: Optional[int] = None
    if raw_year is not None:
        try:
            year = int(raw_year)
        except (TypeError, ValueError):
            logger.warning(
                f"  ⚠ Could not parse year value '{raw_year}', setting to null."
            )
            year = None

    # --- needs_year_filter ---
    needs_year_filter: bool = bool(data.get("needs_year_filter", False))

    # If year is null, year filtering is meaningless — override to False
    if year is None and needs_year_filter:
        logger.warning(
            "  ⚠ needs_year_filter=true but year=null; "
            "overriding needs_year_filter to false."
        )
        needs_year_filter = False

    # --- queries ---
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list) or len(raw_queries) == 0:
        logger.warning("  ⚠ No queries returned; falling back to original question.")
        raw_queries = [original_question]

    # Clamp to max 6 queries
    if len(raw_queries) > 6:
        logger.warning(
            f"  ⚠ Gemini returned {len(raw_queries)} queries; clamping to 6."
        )
        raw_queries = raw_queries[:6]

    # Ensure original question is always first
    queries: List[str] = [q.strip() for q in raw_queries if q.strip()]
    if not queries:
        queries = [original_question]
    elif queries[0].lower() != original_question.lower():
        # Prepend original question and remove any duplicate later in list
        deduped = [original_question] + [
            q for q in queries if q.lower() != original_question.lower()
        ]
        queries = deduped[:6]

    return QueryPlan(
        topic=topic,
        year=year,
        needs_year_filter=needs_year_filter,
        queries=queries,
    )


def _log_query_plan(plan: QueryPlan) -> None:
    """Emit detailed structured logs for the query plan."""
    logger.info("─" * 60)
    logger.info("QUERY PLAN RESULTS")
    logger.info("─" * 60)
    logger.info(f"  Detected Topic   : {plan.topic}")
    logger.info(f"  Detected Year    : {plan.year if plan.year else 'None (no year filter)'}")
    logger.info(f"  Needs Year Filter: {plan.needs_year_filter}")
    logger.info(f"  Total Queries    : {len(plan.queries)}")
    logger.info("  Generated Queries:")
    for i, q in enumerate(plan.queries, 1):
        logger.info(f"    {i}. {q}")
    logger.info("─" * 60)
