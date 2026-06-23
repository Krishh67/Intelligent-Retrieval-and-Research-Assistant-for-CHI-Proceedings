"""
9_rag_pipeline.py

Production-ready RAG Pipeline for CHI Research Assistant
Combines intelligent retrieval planning, multi-query dense retrieval,
metadata-aware filtering, and Gemini answer generation.

Architecture:
═══════════════════════════════════════════════════════════════════
         User Query
              │
              ▼
  ┌───────────────────────┐
  │   Query Planner       │  ← ONE Gemini call
  │  (analyze_and_expand) │
  │  • Detects year/topic │
  │  • Detects section    │
  │  • Generates 2–6 sub- │
  │    queries            │
  └───────────┬───────────┘
              │  QueryPlan (JSON)
              ▼
  ┌───────────────────────┐
  │  Year-Boosted Queries │  ← When year != null, augment every
  │  (pre-retrieval)      │    query with "CHI {year}" suffix
  └───────────┬───────────┘
              │  Boosted queries
              ▼
  ┌───────────────────────┐
  │  Multi-Query FAISS    │  ← N parallel retrieval calls
  │  Retrieval            │    (one per boosted query)
  │  retrieve_papers()×N  │
  └───────────┬───────────┘
              │  All raw results
              ▼
  ┌───────────────────────┐
  │  Cross-Query Merge    │  ← Keep highest score per paper_id
  │  & Deduplication      │
  └───────────┬───────────┘
              │  Deduplicated, ranked list
              ▼
  ┌───────────────────────┐
  │  Metadata Year Filter │  ← Only when needs_year_filter=True
  │  (post-retrieval)     │    Falls back to unfiltered if <3 remain
  └───────────┬───────────┘
              │  Year-filtered list
              ▼
  ┌───────────────────────┐
  │  Section Filter       │  ← Only when section != null
  │  (post-retrieval)     │    Falls back to unfiltered if <3 remain
  └───────────┬───────────┘
              │  Top 8 papers
              ▼
  ┌───────────────────────┐
  │  Context Builder      │
  └───────────┬───────────┘
              │
              ▼
  ┌───────────────────────┐
  │  Gemini 2.5 Flash     │  ← Answer generation call
  │  Answer Generation    │
  └───────────┬───────────┘
              │
              ▼
           Answer
═══════════════════════════════════════════════════════════════════

Key rules:
  - retrieval_v2.py is NOT modified.
  - FAISS index is NOT rebuilt.
  - Embeddings are NOT regenerated.
  - Metadata filtering happens AFTER retrieval, NOT inside FAISS.
  - Year boosting happens BEFORE retrieval (query augmentation only).
  - Total Gemini calls per user question: 2 (planner + answer).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from retrieval_v2 import CHIRetriever, RetrievalResult
from query_planner import QueryPlan, analyze_and_expand_query
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RAGConfig:
    """
    Configuration for the CHI Research Assistant RAG Pipeline.

    Attributes:
        model_name             : Gemini model for answer generation.
        planner_model          : Gemini model for query planning (can be same or lighter).
        temperature            : LLM temperature for answer generation (0.0–1.0).
        max_output_tokens      : Maximum tokens in the generated answer.
        faiss_path             : Path to FAISS index file.
        metadata_path          : Path to metadata pickle file.
        chunks_path            : Path to chunks JSONL file.
        per_query_k            : Papers to return per individual retrieval query.
        per_query_retrieve_k   : Chunks retrieved from FAISS per query (before dedup).
        final_top_k            : Maximum papers kept after merging all queries.
        fallback_min_papers    : Minimum papers required after year-filtering
                                 before falling back to unfiltered results.
        fallback_min_section   : Minimum papers required after section-filtering
                                 before falling back to unfiltered results.
    """
    model_name: str = "gemini-2.5-flash"
    planner_model: str = "gemini-2.5-flash"
    temperature: float = 0.6
    max_output_tokens: int = 2048
    faiss_path: Path = Path("embedding/faiss.index")
    metadata_path: Path = Path("embedding/metadata.pkl")
    chunks_path: Path = Path("embedding/chunks.jsonl")
    per_query_k: int = 5
    per_query_retrieve_k: int = 20
    final_top_k: int = 8
    fallback_min_papers: int = 3


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    """
    Structured output from the full RAG pipeline.

    Attributes:
        question              : Original user question.
        answer                : Generated answer from Gemini.
        papers                : Final list of retrieved papers used in the answer.
        context_length        : Approximate token count of the context window.
        model                 : LLM model used for generation.
        plan                  : The QueryPlan produced by the planner.
        total_raw_papers      : Total papers before deduplication/filtering.
        after_dedup           : Papers remaining after cross-query deduplication.
        after_filter          : Papers remaining after metadata year filter.
        filter_applied        : Whether the year filter was actually applied.
        filter_fallback       : True if year filter was skipped due to too few results.
    """
    question: str
    answer: str
    papers: List[RetrievalResult]
    context_length: int
    model: str
    plan: QueryPlan
    total_raw_papers: int
    after_dedup: int
    after_filter: int
    filter_applied: bool
    filter_fallback: bool


# ---------------------------------------------------------------------------
# Main assistant class
# ---------------------------------------------------------------------------


class CHIResearchAssistant:
    """
    Production-ready CHI Research Assistant with intelligent retrieval planning.

    Retrieval workflow per question:
      1.  analyze_and_expand_query()   → QueryPlan  (1 Gemini call)
      2.  Year boosting (pre-retrieval)→ augment queries with "CHI {year}" when year detected
      3.  Multi-query FAISS retrieval  → N × retrieve_papers() (one per boosted query)
      4.  Cross-query merge + dedup    → highest-score paper per paper_id
      5.  Optional year filter         → discard off-year papers (with fallback)
      6.  build_context()              → formatted context string
      7.  generate_answer()            → Gemini answer (1 Gemini call)

    Total Gemini calls per user question: 2.
    """

    SYSTEM_PROMPT = """\
You are an expert research assistant specialized in Human-Computer Interaction \
(HCI) and the CHI conference.

Your role:
- Answer questions ONLY using the provided research papers
- Synthesize information across multiple papers when relevant
- Always cite paper titles and publication years
- Help researchers understand existing work in HCI

Critical guidelines:
1. If the provided papers don't contain information to answer the question, \
   explicitly say: "I couldn't find relevant information in the provided papers."
2. Never hallucinate or make up papers, findings, or results
3. Always cite which paper each piece of information comes from
4. When multiple papers discuss the same topic, synthesize their findings
5. Be honest about limitations: if papers only partially address the question, say so
6. Prefer direct evidence from papers over speculation
7. For methodological questions, reference the methods used in papers
8. For design questions, ground answers in described design approaches

Output format:
- Start with a direct answer to the question
- Reference specific papers with year
- Highlight key findings from multiple papers if relevant
- Acknowledge gaps or limitations in the literature

Remember: Your credibility depends on accuracy and honesty. If unsure, say so."""

    def __init__(self, config: RAGConfig):
        """
        Initialize the RAG pipeline.

        Args:
            config: RAGConfig with model and path settings.

        Raises:
            ValueError : If GOOGLE_API_KEY is not set.
            RuntimeError: If the retriever fails to load.
        """
        self.config = config

        # ── API Key ─────────────────────────────────────────────────────────
        # Hard-coded key kept here for parity with original file.
        # IMPORTANT: In production, load from environment / secrets manager.
        
        api_key = os.getenv("GOOGLE_API_KEY2")
        if not api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY not set. "
                "Set it via: os.environ['GOOGLE_API_KEY'] = 'your_key'"
            )

        logger.info(f"Configuring Gemini API — model: {config.model_name}")
        genai.configure(api_key=api_key)

        # ── Retriever ────────────────────────────────────────────────────────
        logger.info("Initializing retrieval layer...")
        try:
            self.retriever = CHIRetriever(
                faiss_path=config.faiss_path,
                metadata_path=config.metadata_path,
                chunks_path=config.chunks_path,
            )
            logger.info("✓ Retriever initialized successfully")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize retriever: {exc}") from exc

    # =========================================================================
    # Part 1 — Query Planner (delegates to query_planner.py)
    # =========================================================================

    def plan_query(self, question: str) -> QueryPlan:
        """
        Run the intelligent query planner (ONE Gemini call).

        Wraps analyze_and_expand_query() to integrate cleanly with the
        assistant's existing configuration.

        Args:
            question: Raw user question.

        Returns:
            QueryPlan with topic, year, section, filter flag, and queries.
        """
        return analyze_and_expand_query(
            question=question,
            model_name=self.config.planner_model,
        )

    # =========================================================================
    # Part 2 — Multi-Query Retrieval
    # =========================================================================

    def multi_query_retrieve(
        self,
        plan: QueryPlan
    ) -> Tuple[List[RetrievalResult], int]:
        """
        Execute retrieval for every query in the plan, then merge results.

        Algorithm:
          If plan.year is not None:
            • Augment each query with "CHI {year}" suffix (year boosting)
            • This biases semantic retrieval toward the requested year
            • No FAISS changes, no embedding changes — query text only

          For each (boosted) query q in plan.queries:
            • Call retriever.retrieve_papers(q, k, retrieve_k)
            • Collect all returned RetrievalResult objects

          Merge all results across queries:
            • Deduplicate by paper_id — keep highest score
            • Sort descending by score
            • Keep top final_top_k papers

        Args:
            plan: QueryPlan produced by plan_query().

        Returns:
            Merged, deduplicated list of RetrievalResult sorted by score desc.
        """
        logger.info("─" * 60)
        logger.info("MULTI-QUERY RETRIEVAL")
        logger.info("─" * 60)

        # ── Task 2: Year-aware query boosting (pre-retrieval) ─────────────────
        boosted_queries = self._apply_year_boosting(plan)

        # ── Per-query retrieval ───────────────────────────────────────────────
        all_results: List[RetrievalResult] = []
        for i, query in enumerate(boosted_queries, 1):
            logger.info(f"  Running query {i}/{len(boosted_queries)}: {query!r}")
            try:
                results = self.retriever.retrieve_papers(
                    query=query,
                    k=self.config.per_query_k,
                    retrieve_k=self.config.per_query_retrieve_k,
                )
                logger.info(f"  → Query {i} returned {len(results)} papers")
                all_results.extend(results)
            except Exception as exc:
                logger.warning(
                    f"  ⚠ Query {i} failed ({exc}); skipping."
                )

        total_raw = len(all_results)
        logger.info(f"\n  Merged Papers (before dedup) : {total_raw}")

        # ── Cross-query deduplication ─────────────────────────────────────────
        deduped = self._deduplicate_results(all_results)
        logger.info(f"  After Deduplication          : {len(deduped)}")

        return deduped, total_raw

    # =========================================================================
    # Task 2 helper — Year-Aware Query Boosting
    # =========================================================================

    def _apply_year_boosting(self, plan: QueryPlan) -> List[str]:
        """
        Augment retrieval queries with the detected year BEFORE embedding.

        When the planner detects a specific year (e.g. 2024), every generated
        query is suffixed with "CHI {year}" so the semantic embedding is
        biased toward year-specific vocabulary in the index.

        This is a purely textual transformation — FAISS, embeddings, and
        metadata are never touched.

        Args:
            plan: QueryPlan with optional year and list of queries.

        Returns:
            List of (possibly boosted) query strings, same length as plan.queries.
        """
        if plan.year is None:
            logger.info("  Year Boosting            : Not applied (no year detected)")
            return list(plan.queries)

        logger.info("─" * 60)
        logger.info("YEAR-AWARE QUERY BOOSTING")
        logger.info("─" * 60)
        logger.info(f"  Detected Year            : {plan.year}")

        boosted: List[str] = []
        for original_query in plan.queries:
            boosted_query = f"{original_query} CHI {plan.year}"
            logger.info(f"  Original Query           : {original_query}")
            logger.info(f"  Boosted Query            : {boosted_query}")
            boosted.append(boosted_query)

        return boosted

    def _deduplicate_results(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Merge results across queries keeping the highest score per paper_id.

        Args:
            results: Raw list of RetrievalResult (may have duplicates).

        Returns:
            Deduplicated list sorted by score descending.
        """
        best: Dict[str, RetrievalResult] = {}
        for r in results:
            if r.paper_id not in best or r.score > best[r.paper_id].score:
                best[r.paper_id] = r

        merged = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return merged

    # =========================================================================
    # Part 3 — Metadata Year Filtering
    # =========================================================================

    def apply_year_filter(
        self,
        papers: List[RetrievalResult],
        plan: QueryPlan,
    ) -> Tuple[List[RetrievalResult], bool, bool]:
        """
        Apply year-based metadata filtering after retrieval.

        Filtering is performed POST-FAISS — the index is never modified.

        Logic:
          IF needs_year_filter=True AND year is not None:
            filter papers where paper.year == plan.year
            IF len(filtered) >= fallback_min_papers:
              use filtered list (capped at final_top_k)
            ELSE:
              fall back to unfiltered list (fallback_min_papers not met)
          ELSE:
            no filter — return top final_top_k papers unchanged

        Args:
            papers: Deduplicated, sorted list of RetrievalResult.
            plan  : QueryPlan with filter parameters.

        Returns:
            (final_papers, filter_applied, filter_fallback) tuple where:
              final_papers   : The papers to use for context building.
              filter_applied : True if year filter was successfully applied.
              filter_fallback: True if filter had too few results → used unfiltered.
        """
        logger.info("─" * 60)
        logger.info("METADATA YEAR FILTERING")
        logger.info("─" * 60)

        top_k = self.config.final_top_k
        fallback_min = self.config.fallback_min_papers

        if not plan.needs_year_filter or plan.year is None:
            logger.info(
                f"  Year filter not requested → returning top {top_k} papers."
            )
            final = papers[:top_k]
            logger.info(f"  Final Papers             : {len(final)}")
            return final, False, False

        # Filter by year
        logger.info(f"  Filtering for year       : {plan.year}")
        filtered = [p for p in papers if p.year == plan.year]
        logger.info(f"  After Year Filter        : {len(filtered)}")

        if len(filtered) >= fallback_min:
            final = filtered[:top_k]
            logger.info(
                f"  ✓ Year filter accepted   : {len(final)} papers retained"
            )
            return final, True, False
        else:
            # Fallback: year filter would leave too few papers
            logger.warning(
                f"  ⚠ Only {len(filtered)} papers passed year filter "
                f"(threshold: {fallback_min}). "
                f"Falling back to unfiltered results."
            )
            final = papers[:top_k]
            logger.info(
                f"  Fallback Final Papers    : {len(final)}"
            )
            return final, False, True

    

    # =========================================================================
    # Context Builder (unchanged from original)
    # =========================================================================

    def build_context(self, papers: List[RetrievalResult]) -> Tuple[str, int]:
        """
        Build context string from retrieved papers.

        Args:
            papers: Final list of RetrievalResult objects.

        Returns:
            (context_string, estimated_token_count) tuple.

        Context block format:
            ────────────────────────────────────────────────────────────
            PAPER N
            ────────────────────────────────────────────────────────────
            Paper ID : ...
            Year     : ...
            Title    : ...
            Section  : ...

            <chunk text>

        Token estimate: ~1.3 characters per token (OpenAI approximation).
        """
        context_parts = []
        for i, paper in enumerate(papers, 1):
            sep = "─" * 80
            block = (
                f"{sep}\n"
                f"PAPER {i}\n"
                f"{sep}\n"
                f"Paper ID:  {paper.paper_id}\n"
                f"Year:      {paper.year}\n"
                f"Title:     {paper.title}\n"
                f"Section:   {paper.section}\n\n"
                f"{paper.chunk_text}\n"
            )
            context_parts.append(block)

        context = "\n".join(context_parts)
        context += f"\n{'─' * 80}\n"

        estimated_tokens = int(len(context) / 1.3)
        logger.info(
            f"✓ Built context from {len(papers)} papers (~{estimated_tokens} tokens)"
        )
        return context, estimated_tokens

    # =========================================================================
    # Answer Generation (unchanged from original)
    # =========================================================================

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate a grounded answer using Gemini.

        Args:
            question: User question.
            context : Context built from retrieved papers.

        Returns:
            Answer string from Gemini.

        Raises:
            RuntimeError: If the Gemini API call fails.
        """
        logger.info(f"Sending context to {self.config.model_name} for generation...")
        try:
            model = genai.GenerativeModel(
                model_name=self.config.model_name,
                system_instruction=self.SYSTEM_PROMPT,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                ),
            )
            prompt = (
                "Based on the following research papers from the CHI conference, "
                "answer this question:\n\n"
                f"QUESTION: {question}\n\n"
                f"RESEARCH PAPERS:\n{context}\n\n"
                "Please provide a comprehensive answer grounded in the provided papers."
            )
            response = model.generate_content(prompt)
            logger.info("✓ Answer generated successfully")
            return response.text
        except Exception as exc:
            logger.error(f"Gemini API error: {exc}", exc_info=True)
            raise RuntimeError(f"Failed to generate answer: {exc}") from exc

    # =========================================================================
    # Part 4 — Logging summary
    # =========================================================================

    def _log_pipeline_summary(
        self,
        plan: QueryPlan,
        total_raw: int,
        after_dedup: int,
        after_filter: int,
        final_count: int,
        filter_applied: bool,
        filter_fallback: bool,
    ) -> None:
        """
        Emit a consolidated pipeline summary log block.
        """
        logger.info("═" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("═" * 60)
        logger.info(f"  Detected Topic   : {plan.topic}")
        logger.info(
            f"  Detected Year    : {plan.year if plan.year else 'None'}"
        )
        logger.info(f"  Needs Filter     : {plan.needs_year_filter}")
        logger.info(f"  Generated Queries: {len(plan.queries)}")
        for i, q in enumerate(plan.queries, 1):
            logger.info(f"    {i}. {q}")
        logger.info("  ─" * 30)
        logger.info(f"  Merged Papers    : {total_raw}")
        logger.info(f"  After Dedup      : {after_dedup}")
        if filter_applied:
            logger.info(f"  After Year Filter: {after_filter}  ✓ (year={plan.year})")
        elif filter_fallback:
            logger.info(
                f"  After Year Filter: {after_filter}  ⚠ FALLBACK — too few results"
            )
        else:
            logger.info(f"  Year Filter      : Not applied")
        logger.info(f"  Final Papers     : {final_count}")
        logger.info("═" * 60)

    # =========================================================================
    # Part 5 — Main ask() entry point (updated workflow)
    # =========================================================================

    def ask(self, question: str) -> "RAGResponse":
        """
        Complete RAG pipeline: plan → retrieve → filter → build → generate.

        Updated workflow:
          1. analyze_and_expand_query()  → QueryPlan       (1 Gemini call)
          2. multi_query_retrieve()      → year-boosted queries → merged papers
          3. apply_year_filter()         → year-filtered papers
          4. apply_section_filter()      → section-filtered papers
          5. build_context()             → context string
          6. generate_answer()           → answer           (1 Gemini call)

        Args:
            question: User's natural language question.

        Returns:
            RAGResponse with answer, papers, plan, and pipeline statistics.

        Raises:
            ValueError : If no papers are retrieved.
            RuntimeError: If generation fails.
        """
        logger.info("\n" + "═" * 70)
        logger.info(f"QUESTION: {question!r}")
        logger.info("═" * 70 + "\n")

        # ── Step 1: Query Planning ────────────────────────────────────────────
        plan = self.plan_query(question)

        # ── Step 2: Multi-Query Retrieval (with year boosting inside) ─────────
        deduped_papers, total_raw = self.multi_query_retrieve(plan)
        after_dedup = len(deduped_papers)

        if not deduped_papers:
            raise ValueError("No papers retrieved. Try a different query.")

        # ── Step 3: Metadata Year Filtering ───────────────────────────────────
        year_filtered_papers, filter_applied, filter_fallback = self.apply_year_filter(
            deduped_papers, plan
        )
        after_filter = len(year_filtered_papers)

        # ── Step 4: Section Filtering ─────────────────────────────────────────
        # in future
        final_papers = year_filtered_papers
        


        # ── Step 5: Log pipeline summary ──────────────────────────────────────
        self._log_pipeline_summary(
            plan=plan,
            total_raw=total_raw,
            after_dedup=after_dedup,
            after_filter=after_filter,
            final_count=len(final_papers),
            filter_applied=filter_applied,
            filter_fallback=filter_fallback
        )

        # ── Step 6: Build context ─────────────────────────────────────────────
        context, token_count = self.build_context(final_papers)

        # ── Step 7: Generate answer ───────────────────────────────────────────
        answer = self.generate_answer(question, context)

        logger.info("✓ RAG pipeline completed\n")

        return RAGResponse(
            question=question,
            answer=answer,
            papers=final_papers,
            context_length=token_count,
            model=self.config.model_name,
            plan=plan,
            total_raw_papers=total_raw,
            after_dedup=after_dedup,
            after_filter=after_filter,
            filter_applied=filter_applied,
            filter_fallback=filter_fallback,
           
        )


# ---------------------------------------------------------------------------
# Convenience initializer
# ---------------------------------------------------------------------------


def load_retriever(config: Optional[RAGConfig] = None) -> CHIResearchAssistant:
    """
    Convenience function to initialize the RAG pipeline with default config.

    Usage:
        assistant = load_retriever()
        response  = assistant.ask("Your question here")

    Args:
        config: Optional RAGConfig (uses defaults if None).

    Returns:
        Initialized CHIResearchAssistant instance.
    """
    if config is None:
        config = RAGConfig()
    return CHIResearchAssistant(config)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def format_response(response: RAGResponse) -> str:
    """
    Format a RAGResponse for human-readable CLI display.

    Args:
        response: RAGResponse object.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("ANSWER")
    lines.append("=" * 90)
    lines.append(response.answer)

    lines.append("\n" + "=" * 90)
    lines.append("KEY PAPERS")
    lines.append("=" * 90)
    for i, paper in enumerate(response.papers, 1):
        lines.append(f"  {i}. {paper.title} ({paper.year})  [score: {paper.score:.4f}]")

    lines.append("\n" + "=" * 90)
    lines.append("RETRIEVAL METADATA")
    lines.append("=" * 90)
    lines.append(f"  Detected Topic     : {response.plan.topic}")
    lines.append(f"  Detected Year      : {response.plan.year}")
    # Year filter
    lines.append(f"  Year Filter Applied: {response.filter_applied}")
    if response.filter_fallback:
        lines.append("  ⚠ Year filter fell back to unfiltered results (too few papers)")
   
    lines.append(f"  Generated Queries  : {len(response.plan.queries)}")
    for i, q in enumerate(response.plan.queries, 1):
        lines.append(f"    {i}. {q}")
    lines.append(f"\n  Total Merged Papers    : {response.total_raw_papers}")
    lines.append(f"  After Deduplication    : {response.after_dedup}")
    lines.append(f"  After Year Filter      : {response.after_filter}")
    lines.append(f"  Final Papers           : {len(response.papers)}")

    lines.append("\n" + "=" * 90)
    lines.append("GENERATION METADATA")
    lines.append("=" * 90)
    lines.append(f"  Model             : {response.model}")
    lines.append(f"  Context Size      : ~{response.context_length} tokens")
    lines.append("=" * 90 + "\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------


def interactive_cli() -> None:
    """
    Interactive command-line interface for CHI Research Assistant.

    Usage:
        python 9_rag_pipeline.py

    Then enter questions at the prompt. Type 'exit' to quit.
    """
    print("\n" + "=" * 90)
    print("CHI RESEARCH ASSISTANT — INTELLIGENT RAG PIPELINE v3.0")
    print("=" * 90)
    print("\nInitializing...")

    try:
        assistant = load_retriever()
    except ValueError as exc:
        print(f"\n❌ Error: {exc}")
        print("\nTo fix, set your API key:")
        print("  import os")
        print("  os.environ['GOOGLE_API_KEY'] = 'your_api_key_here'")
        return
    except RuntimeError as exc:
        print(f"\n❌ Error: {exc}")
        return

    print("\n" + "=" * 90)
    print("System Configuration:")
    print(f"  • Answer Model      : {assistant.config.model_name}")
    print(f"  • Planner Model     : {assistant.config.planner_model}")
    print(f"  • Temperature       : {assistant.config.temperature}")
    print(f"  • Max Output Tokens : {assistant.config.max_output_tokens}")
    print("\nRetrieval Configuration:")
    print(f"  • Per-Query k       : {assistant.config.per_query_k} papers")
    print(f"  • Per-Query FAISS k : {assistant.config.per_query_retrieve_k} chunks")
    print(f"  • Final Top k       : {assistant.config.final_top_k} papers")
    print(f"  • Fallback Min      : {assistant.config.fallback_min_papers} papers")
    print(f"  • Total Chunks      : {len(assistant.retriever.chunks):,}")
    print(f"  • FAISS Vectors     : {assistant.retriever.faiss_index.ntotal:,}")
    print(f"  • Embedding Model   : {assistant.retriever.model_name}")
    print("\nDataset Statistics:")
    print("  • CHI Papers        : 2,635")
    print("  • Years             : 2021, 2023, 2024")
    print("  • Sections          : 21,802")
    print("=" * 90)
    print("\nType 'exit' to quit.\n")

    while True:
        try:
            question = input("Enter your question: ").strip()

            if question.lower() == "exit":
                print("\nThank you for using CHI Research Assistant!")
                break

            if not question:
                print("Please enter a question.\n")
                continue

            response = assistant.ask(question)
            print(format_response(response))

        except ValueError as exc:
            logger.error(f"Validation error: {exc}")
            print(f"\n⚠ {exc}\n")
        except RuntimeError as exc:
            logger.error(f"Runtime error: {exc}")
            print(f"\n❌ {exc}\n")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}", exc_info=True)
            print(f"\n❌ Unexpected error: {exc}\n")


def main() -> None:
    """Entry point for the RAG pipeline."""
    interactive_cli()


if __name__ == "__main__":
    main()
