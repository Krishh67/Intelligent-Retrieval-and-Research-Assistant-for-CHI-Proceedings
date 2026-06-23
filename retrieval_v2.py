"""
retrieval_v2.py

Production-ready Retrieval Layer for CHI RAG System
Retrieves the best papers (not chunks) for a given query

Features:
- FAISS-based dense retrieval with deduplication
- Automatic paper-level deduplication to reduce redundancy
- GPU acceleration if available
- Interactive CLI for testing
- Full type hints and comprehensive logging
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import warnings

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class RetrievalResult:
    """
    Represents a single paper retrieved from the RAG system.

    Attributes:
        paper_id: Unique identifier for the paper
        year: Publication year
        title: Paper title
        section: Section name within the paper
        score: Similarity score from FAISS (higher = more relevant)
        chunk_text: Actual text content of the retrieved chunk
    """
    paper_id: str
    year: int
    title: str
    section: str
    score: float
    chunk_text: str


class CHIRetriever:
    """
    Production-ready retrieval system for the CHI RAG dataset.

    Design Philosophy:
    - FAISS enables fast nearest-neighbor search (O(n) instead of brute force)
    - Normalized embeddings + IndexFlatIP use cosine similarity (efficient for semantic search)
    - Deduplication prevents context window waste (multiple chunks from same paper)
    - Top-5 papers provide diverse, complementary perspectives on the query
    """

    def __init__(
        self,
        faiss_path: Path,
        metadata_path: Path,
        chunks_path: Path,
        model_name: str = "BAAI/bge-large-en-v1.5"
    ):
        """
        Initialize the retriever with FAISS index and metadata.

        Args:
            faiss_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            chunks_path: Path to chunks JSONL file
            model_name: HuggingFace model ID for embeddings
        """
        self.faiss_path = Path(faiss_path)
        self.metadata_path = Path(metadata_path)
        self.chunks_path = Path(chunks_path)
        self.model_name = model_name

        # Load components
        self.embedding_model = self._load_embedding_model()
        self.faiss_index = self._load_faiss_index()
        self.metadata = self._load_metadata()
        self.chunks = self._load_chunks()

        logger.info(f"✓ Retriever initialized with {len(self.chunks)} chunks")

    def _load_embedding_model(self) -> SentenceTransformer:
        """
        Load the sentence transformer model for query embeddings.

        Why BAAI/bge-large-en-v1.5?
        - Optimized for semantic search (not just sentence similarity)
        - Large model (335M params) captures fine-grained semantic nuances
        - Supports 512-token input (good for paper titles + abstracts)
        - Works with GPU acceleration for speed
        - Trained on 400M+ query-passage pairs from academic sources
        """
        logger.info(f"Loading embedding model: {self.model_name}")
        model = SentenceTransformer(self.model_name)

        # Detect GPU availability
        if model.device.type == 'cuda':
            logger.info(f"✓ Using GPU: {model.device}")
        else:
            logger.info("⚠ GPU not available, using CPU (slower)")

        return model

    def _load_faiss_index(self) -> faiss.Index:
        """
        Load the FAISS index.

        Why IndexFlatIP?
        - IP (Inner Product) with normalized vectors = cosine similarity
        - Normalized embeddings ensure scores are in [0, 1] range (normalized)
        - Fast exact search, no approximation loss
        - Simple and reliable for semantic search up to 10M vectors
        - Scales to our 75K chunks efficiently
        """
        logger.info(f"Loading FAISS index from {self.faiss_path}")
        index = faiss.read_index(str(self.faiss_path))
        logger.info(f"✓ Loaded FAISS index with {index.ntotal} vectors")
        return index

    def _load_metadata(self) -> dict:
        """
        Load metadata (chunk_idx -> {paper_id, year, title, section}).

        Returns:
            Dictionary mapping chunk index to metadata
        """
        logger.info(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"✓ Loaded metadata for {len(metadata)} chunks")
        return metadata

    def _load_chunks(self) -> List[str]:
        """
        Load chunk texts from JSONL file.

        Returns:
            List of chunk text strings indexed by chunk_idx
        """
        logger.info(f"Loading chunks from {self.chunks_path}")
        chunks = []
        with open(
            self.chunks_path,
            'r',
            encoding='utf-8'
        ) as f:
            for line in f:  
                chunk_data = json.loads(line)
                chunks.append(chunk_data['text'])
        logger.info(f"✓ Loaded {len(chunks)} chunks")
        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a natural language query to an embedding.

        Args:
            query: Natural language question

        Returns:
            Normalized embedding vector (shape: (768,) for bge-large-en-v1.5)

        Why normalization?
        - FAISS IndexFlatIP computes dot product between vectors
        - With L2-normalized vectors: dot_product(u, v) = cosine_similarity(u, v)
        - Ensures scores are comparable across different queries
        - Required to match how embeddings were stored in the index
        """
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # L2 normalization: x / ||x||
        norm = np.linalg.norm(embedding)
        embedding = embedding / (norm + 1e-8)

        return embedding.astype(np.float32)

    def retrieve_chunks(
        self,
        query_embedding: np.ndarray,
        k: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k most similar chunks from FAISS.

        Args:
            query_embedding: Normalized query embedding (shape: (768,))
            k: Number of chunks to retrieve (default: 20)

        Returns:
            scores: Similarity scores (shape: (k,))
            indices: Indices of retrieved chunks (shape: (k,))

        Why retrieve 20 chunks?
        - Provides diversity before deduplication
        - Even with duplicates, likely to get 5+ unique papers
        - Balances quality (more results) vs speed (not too many)
        - 20 is a good empirical choice for paper-level retrieval
        """
        # FAISS expects 2D input (batch of queries)
        query_embedding = query_embedding.reshape(1, -1)

        # Search: returns (scores, indices) for each query
        scores, indices = self.faiss_index.search(query_embedding, k)
        scores = scores[0]  # Remove batch dimension
        indices = indices[0]

        logger.info(f"✓ Retrieved {len(indices)} chunks from FAISS")
        return scores, indices

    def deduplicate_by_paper(
        self,
        scores: np.ndarray,
        indices: np.ndarray
    ) -> List[Tuple[str, float, int]]:
        """
        Keep only the highest-scoring chunk from each paper.

        Args:
            scores: Similarity scores from FAISS (shape: (k,))
            indices: Chunk indices from FAISS (shape: (k,))

        Returns:
            List of (paper_id, score, chunk_idx) tuples, deduplicated by paper

        Why deduplication?
        - Multiple sections from the same paper are often highly similar
        - Without dedup: top-5 might be [Jigsaw, Jigsaw, Jigsaw, Jigsaw, Jigsaw]
        - Wastes token budget to include 5 chunks from the same paper
        - Returns diverse papers instead, providing complementary views

        Example:
            Input chunks:  [paper_12, paper_12, paper_12, paper_48, paper_91]
            Scores:        [0.95,     0.93,     0.91,     0.87,     0.85]
            Output papers: [(paper_12, 0.95), (paper_48, 0.87), (paper_91, 0.85)]
            - Kept best chunk from paper_12 (score 0.95)
            - Removed duplicate chunks from paper_12
        """
        paper_scores = {}

        for score, chunk_idx in zip(scores, indices):
            chunk_idx = int(chunk_idx)
            metadata = self.metadata[chunk_idx]
            paper_id = metadata['paper_id']

            # Keep only the highest scoring chunk per paper
            if paper_id not in paper_scores:
                paper_scores[paper_id] = (score, chunk_idx)
            else:
                existing_score, _ = paper_scores[paper_id]
                if score > existing_score:
                    paper_scores[paper_id] = (score, chunk_idx)

        # Sort by score descending (best papers first)
        results = [
            (paper_id, score, chunk_idx)
            for paper_id, (score, chunk_idx) in paper_scores.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"✓ Deduplicated to {len(results)} unique papers from {len(indices)} chunks")
        return results

    def retrieve_papers(
        self,
        query: str,
        k: int = 5,
        retrieve_k: int = 20
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-k best papers for a query.

        Args:
            query: Natural language question
            k: Number of papers to return (default: 5)
            retrieve_k: Number of chunks to retrieve before dedup (default: 20)

        Returns:
            List of RetrievalResult objects sorted by relevance

        Workflow:
            1. Embed query with BAAI/bge-large-en-v1.5
            2. Retrieve 20 chunks from FAISS (fast nearest-neighbor)
            3. Deduplicate by paper_id (keep best chunk per paper)
            4. Return top-5 papers with full metadata
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing query: '{query}'")
        logger.info(f"{'='*60}")

        # Step 1: Embed query
        query_embedding = self.embed_query(query)

        # Step 2: Retrieve chunks
        scores, indices = self.retrieve_chunks(query_embedding, k=retrieve_k)

        # Step 3: Deduplicate by paper
        deduplicated = self.deduplicate_by_paper(scores, indices)

        # Step 4: Build results
        results = []
        for paper_id, score, chunk_idx in deduplicated[:k]:
            chunk_idx = int(chunk_idx)
            metadata = self.metadata[chunk_idx]
            chunk_text = self.chunks[chunk_idx]

            result = RetrievalResult(
                paper_id=paper_id,
                year=metadata['year'],
                title=metadata['title'],
                section=metadata['section'],
                score=float(score),
                chunk_text=chunk_text
            )
            results.append(result)

        logger.info(f"✓ Returned {len(results)} papers\n")
        return results


def initialize_retriever(
    base_dir: Path = Path("embedding")
) -> CHIRetriever:
    """
    Convenience function to initialize the retriever with Colab paths.

    Usage in Colab:
        retriever = initialize_retriever()
        results = retriever.retrieve_papers("Your query here")

    Args:
        base_dir: Directory containing faiss.index and metadata.pkl
        chunks_dir: Directory containing chunks.jsonl

    Returns:
        Initialized CHIRetriever instance
    """
    faiss_path = base_dir / "faiss.index"
    metadata_path = base_dir / "metadata.pkl"
    chunks_path ="chunks.jsonl"

    return CHIRetriever(
        faiss_path=faiss_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path
    )


def display_results(results: List[RetrievalResult]) -> None:
    """
    Pretty-print retrieval results for CLI.

    Args:
        results: List of RetrievalResult objects
    """
    print("\n" + "=" * 90)
    print("RETRIEVAL RESULTS")
    print("=" * 90)

    for rank, result in enumerate(results, 1):
        print(f"\n{'─' * 90}")
        print(f"RANK {rank} | Score: {result.score:.4f} | Year: {result.year}")
        print(f"{'─' * 90}")
        print(f"Title:   {result.title}")
        print(f"Section: {result.section}")
        print(f"Paper:   {result.paper_id}")
        print(f"\nText Preview (first 500 characters):")
        print(f"{result.chunk_text[:500]}")
        if len(result.chunk_text) > 500:
            print("...")

    print("\n" + "=" * 90 + "\n")


def interactive_cli() -> None:
    """
    Interactive command-line interface for testing queries.

    Usage:
        python 09_retrieval_v2.py

    Then enter queries at the prompt. Type 'exit' to quit.
    """
    logger.info("Initializing retriever...")
    retriever = initialize_retriever()

    print("\n" + "=" * 90)
    print("CHI RAG RETRIEVAL SYSTEM v2.0")
    print("=" * 90)
    print("\nSystem Configuration:")
    print(f"  • Embedding Model:     {retriever.model_name}")
    print(f"  • Total Chunks:        {len(retriever.chunks):,}")
    print(f"  • FAISS Index Vectors: {retriever.faiss_index.ntotal:,}")
    print(f"  • Index Type:          IndexFlatIP (cosine similarity)")
    print(f"  • Embedding Dim:       768")
    print(f"  • Top Papers:          5")
    print(f"  • Retrieve Chunks:     20 (before dedup)")
    print("\nDataset Statistics:")
    print(f"  • CHI Papers:          2,635")
    print(f"  • Years:               2021, 2023, 2024")
    print(f"  • Sections:            21,802")
    print(f"  • Chunks:              75,817")
    print("\nType 'exit' to quit.\n")

    while True:
        try:
            query = input("Enter query: ").strip()

            if query.lower() == 'exit':
                print("\nExiting...")
                break

            if not query:
                print("Please enter a query.\n")
                continue

            results = retriever.retrieve_papers(query, k=5, retrieve_k=20)
            display_results(results)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    interactive_cli()
