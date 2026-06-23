#!/usr/bin/env python3
"""
06_generate_embeddings.py

Purpose:
--------
Generate dense embeddings for academic paper chunks using a state-of-the-art
semantic search model, build a FAISS index, and prepare the system for
production RAG retrieval.

Key Design Decisions:
---------------------
1. DENSE EMBEDDINGS:
   - Use BAAI/bge-base-en-v1.5 (768-dim, optimized for semantic search)
   - Outperforms older models like BERT on information retrieval tasks
   - Supports both title and section context for better semantic understanding

2. CONTEXTUAL TEXT REPRESENTATION:
   - Embed: "Title: {title}\n\nSection: {section}\n\n{text}"
   - Title provides document context (avoids ambiguity)
   - Section name acts as a semantic anchor (INTRO vs METHODS != same)
   - Preserves full chunk text for semantic richness
   - Excludes metadata (paper_id, year, chunk_id) to focus on semantic content

3. BATCH PROCESSING:
   - Reduces memory overhead vs single-sample processing
   - Leverages GPU/CPU parallelism
   - Default batch_size=64 balances speed and memory
   - Configurable for different hardware constraints

4. SEPARATE METADATA STORAGE:
   - Embeddings stored in .npy format (numpy binary, efficient)
   - Metadata stored in .pkl format (Python pickle, preserves structure)
   - Alignment guaranteed: embeddings[i] ↔ metadata[i]
   - Reduces data duplication and memory overhead
   - Enables efficient streaming and distributed processing

5. FAISS INDEX WITH NORMALIZATION:
   - IndexFlatIP: Inner Product search (equivalent to cosine after normalization)
   - L2 normalization ensures embeddings are unit vectors
   - Cosine similarity = normalized inner product
   - Fast exact search (no approximation loss)
   - Preparation for future GPU acceleration or approximate indices

6. DEVICE DETECTION:
   - Auto-detect CUDA availability
   - SentenceTransformer automatically uses GPU if available
   - FAISS index created on CPU (for portability)
   - Significant speedup on GPU (10-100x depending on hardware)

7. STREAMING LOAD + BATCHING:
   - Generator pattern reads JSONL line-by-line
   - Reduces memory footprint for large corpora
   - Enables future pipeline integration (Spark, Ray, Dask)
   - Suitable for 75k+ chunks (and scalable to 500k+)

8. PROGRESS TRACKING & TIME ESTIMATION:
   - Report every 1000 chunks for visibility
   - Track chunks/second for performance monitoring
   - Estimate remaining time for long-running jobs
   - Helps optimize batch sizes and hardware allocation
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Generator, List, Dict, Any, Tuple
import pickle

# Suppress TensorFlow import and warnings
os.environ['TF_CPP_LOGGING_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for chunks using SentenceTransformers.
    
    Responsibilities:
    - Load semantic search model
    - Prepare text for embedding (combine title, section, text)
    - Generate embeddings in batches
    - Track statistics and progress
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 64,
        device: str = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Number of samples per batch
            device: Force device ("cuda" or "cpu"), auto-detect if None
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Model info
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def prepare_text_for_embedding(
        self,
        title: str,
        section: str,
        text: str
    ) -> str:
        """
        Prepare combined text for embedding.
        
        Format:
            Title: {title}
            
            Section: {section}
            
            {text}
        
        Rationale:
        - Title provides document-level context
        - Section provides content-type context (INTRO vs RESULTS are semantically different)
        - Full text preserves semantic richness
        
        Args:
            title: Paper title
            section: Section name
            text: Chunk text
            
        Returns:
            Combined text ready for embedding
        """
        combined_text = f"Title: {title}\n\nSection: {section}\n\n{text}"
        return combined_text
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Embeddings array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings


class ChunkDataLoader:
    """
    Efficiently loads chunks from JSONL file.
    
    Uses generator pattern to minimize memory footprint.
    """
    
    def __init__(self, jsonl_file: Path):
        """
        Initialize data loader.
        
        Args:
            jsonl_file: Path to chunks.jsonl
        """
        self.jsonl_file = Path(jsonl_file)
    
    def load_chunks(self) -> Generator[Dict[str, Any], None, None]:
        """
        Load chunks from JSONL file.
        
        Yields chunks one at a time (generator pattern).
        
        Yields:
            Individual chunk records
        """
        try:
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        yield chunk
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed JSON at line {line_num}: {e}"
                        )
                        continue
        
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.jsonl_file}")
            raise
    
    def validate_chunk(self, chunk: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate chunk record.
        
        Args:
            chunk: Chunk record
            
        Returns:
            Tuple of (is_valid, reason)
        """
        required_fields = ['chunk_id', 'paper_id', 'year', 'title', 'section', 'text']
        
        # Check required fields
        missing = [f for f in required_fields if f not in chunk]
        if missing:
            return False, f"Missing fields: {missing}"
        
        # Check text is not empty
        if not isinstance(chunk['text'], str) or not chunk['text'].strip():
            return False, "Text is empty or not a string"
        
        # Check metadata types
        if not isinstance(chunk['chunk_id'], int):
            return False, f"chunk_id not int: {type(chunk['chunk_id'])}"
        
        if not isinstance(chunk['paper_id'], int):
            return False, f"paper_id not int: {type(chunk['paper_id'])}"
        
        if not isinstance(chunk['year'], int):
            return False, f"year not int: {type(chunk['year'])}"
        
        return True, ""


class RAGCorpusBuilder:
    """
    Main orchestrator for building RAG corpus:
    1. Load chunks
    2. Generate embeddings
    3. Build FAISS index
    4. Save all artifacts
    """
    
    def __init__(
        self,
        chunks_file: Path,
        output_dir: Path,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 64,
        device: str = None
    ):
        """
        Initialize RAG corpus builder.
        
        Args:
            chunks_file: Path to chunks.jsonl
            output_dir: Directory for output files
            model_name: SentenceTransformer model
            batch_size: Embedding batch size
            device: "cuda" or "cpu"
        """
        self.chunks_file = Path(chunks_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = self.output_dir / 'embeddings.npy'
        self.metadata_file = self.output_dir / 'metadata.pkl'
        self.faiss_index_file = self.output_dir / 'faiss.index'
        
        self.embedding_gen = EmbeddingGenerator(
            model_name=model_name,
            batch_size=batch_size,
            device=device
        )
        
        self.data_loader = ChunkDataLoader(self.chunks_file)
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'chunks_processed': 0,
            'chunks_skipped': 0,
            'start_time': None,
            'end_time': None,
        }
        
        self.all_embeddings = []
        self.all_metadata = []
    
    def build_corpus(self) -> None:
        """
        Main pipeline: load -> embed -> index -> save.
        """
        logger.info("=" * 70)
        logger.info("RAG CORPUS BUILDER")
        logger.info("=" * 70)
        logger.info(f"Input file: {self.chunks_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model: {self.embedding_gen.model_name}")
        logger.info(f"Batch size: {self.embedding_gen.batch_size}")
        logger.info(f"Device: {self.embedding_gen.device}")
        logger.info("=" * 70)
        
        self.stats['start_time'] = time.time()
        
        # Phase 1: Generate embeddings
        logger.info("\n[Phase 1] Generating embeddings...")
        self._generate_embeddings()
        
        # Phase 2: Build FAISS index
        logger.info("\n[Phase 2] Building FAISS index...")
        self._build_faiss_index()
        
        # Phase 3: Save artifacts
        logger.info("\n[Phase 3] Saving artifacts...")
        self._save_embeddings()
        self._save_metadata()
        self._save_faiss_index()
        
        self.stats['end_time'] = time.time()
        
        # Phase 4: Print statistics
        logger.info("\n[Phase 4] Printing statistics...")
        self._print_statistics()
    
    def _generate_embeddings(self) -> None:
        """
        Generate embeddings for all chunks.
        
        Strategy:
        - Load chunks in batches
        - Prepare text (title + section + text)
        - Generate embeddings
        - Store embeddings and metadata separately
        """
        batch_texts = []
        batch_metadata = []
        batch_start_time = time.time()
        
        for chunk in self.data_loader.load_chunks():
            self.stats['total_chunks'] += 1
            
            # Validate chunk
            is_valid, reason = self.data_loader.validate_chunk(chunk)
            if not is_valid:
                logger.warning(
                    f"Skipping chunk {chunk.get('chunk_id', '?')}: {reason}"
                )
                self.stats['chunks_skipped'] += 1
                continue
            
            # Prepare text for embedding
            text_for_embedding = self.embedding_gen.prepare_text_for_embedding(
                title=chunk['title'],
                section=chunk['section'],
                text=chunk['text']
            )
            
            # Collect batch
            batch_texts.append(text_for_embedding)
            batch_metadata.append({
                'chunk_id': chunk['chunk_id'],
                'paper_id': chunk['paper_id'],
                'year': chunk['year'],
                'title': chunk['title'],
                'section': chunk['section'],
                'chunk_index': chunk['chunk_index'],
            })
            
            # Process batch when full
            if len(batch_texts) >= self.embedding_gen.batch_size:
                embeddings = self.embedding_gen.generate_embeddings_batch(batch_texts)
                self.all_embeddings.append(embeddings)
                self.all_metadata.extend(batch_metadata)
                
                self.stats['chunks_processed'] += len(batch_texts)
                batch_texts = []
                batch_metadata = []
                
                # Progress logging every 1000 chunks
                if self.stats['chunks_processed'] % 1000 == 0:
                    elapsed = time.time() - batch_start_time
                    chunks_per_sec = self.stats['chunks_processed'] / elapsed
                    total_elapsed = time.time() - self.stats['start_time']
                    
                    if self.stats['chunks_processed'] > 0:
                        remaining = (
                            (self.stats['total_chunks'] - self.stats['chunks_processed']) 
                            / chunks_per_sec
                        )
                    else:
                        remaining = 0
                    
                    logger.info(
                        f"Progress: {self.stats['chunks_processed']} chunks processed | "
                        f"{chunks_per_sec:.1f} chunks/sec | "
                        f"Elapsed: {total_elapsed:.1f}s | "
                        f"Remaining: {remaining:.1f}s"
                    )
        
        # Process remaining batch
        if batch_texts:
            embeddings = self.embedding_gen.generate_embeddings_batch(batch_texts)
            self.all_embeddings.append(embeddings)
            self.all_metadata.extend(batch_metadata)
            self.stats['chunks_processed'] += len(batch_texts)
        
        # Concatenate all embeddings
        self.embeddings_array = np.vstack(self.all_embeddings)
        
        logger.info(
            f"Embeddings generated: {self.embeddings_array.shape} "
            f"(samples × dimensions)"
        )
        logger.info(f"Chunks skipped: {self.stats['chunks_skipped']}")
    
    def _build_faiss_index(self) -> None:
        """
        Build FAISS index from embeddings.
        
        Strategy:
        - Normalize embeddings (L2 norm)
        - Create IndexFlatIP (Inner Product)
        - Cosine similarity = normalized inner product
        - Fast exact search without approximation
        
        Why normalization:
        - Converts dot product to cosine similarity
        - Improves retrieval quality
        - Required for semantic search
        """
        # Normalize embeddings to unit vectors (L2 norm = 1)
        logger.info("Normalizing embeddings...")
        self.embeddings_array_normalized = self.embeddings_array.copy().astype(np.float32)
        faiss.normalize_L2(self.embeddings_array_normalized)
        
        # Create FAISS index with normalized embeddings
        logger.info(f"Creating FAISS IndexFlatIP...")
        self.index = faiss.IndexFlatIP(self.embeddings_array_normalized.shape[1])
        self.index.add(self.embeddings_array_normalized)
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
    
    def _save_embeddings(self) -> None:
        """
        Save embeddings to .npy file.
        
        Format: (num_chunks, embedding_dim)
        Example: (75817, 768)
        """
        np.save(self.embeddings_file, self.embeddings_array_normalized)
        file_size_mb = self.embeddings_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"Embeddings saved: {self.embeddings_file} "
            f"({file_size_mb:.2f} MB)"
        )
    
    def _save_metadata(self) -> None:
        """
        Save metadata to .pkl file.
        
        Contains: chunk_id, paper_id, year, title, section, chunk_index
        Does NOT contain embeddings (stored separately in .npy)
        Does NOT duplicate chunk text (stored in original chunks.jsonl)
        
        Alignment guaranteed:
        - metadata[i] ↔ embeddings[i]
        """
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.all_metadata, f)
        
        file_size_mb = self.metadata_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"Metadata saved: {self.metadata_file} "
            f"({file_size_mb:.2f} MB)"
        )
    
    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk."""
        faiss.write_index(self.index, str(self.faiss_index_file))
        file_size_mb = self.faiss_index_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"FAISS index saved: {self.faiss_index_file} "
            f"({file_size_mb:.2f} MB)"
        )
    
    def _print_statistics(self) -> None:
        """Print final corpus statistics."""
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "=" * 70)
        print("RAG CORPUS STATISTICS")
        print("=" * 70)
        print(f"Total Chunks Loaded:        {self.stats['total_chunks']}")
        print(f"Total Chunks Processed:     {self.stats['chunks_processed']}")
        print(f"Total Chunks Skipped:       {self.stats['chunks_skipped']}")
        print(f"Embedding Dimension:        {self.embedding_gen.embedding_dim}")
        print(f"Embeddings Shape:           {self.embeddings_array.shape}")
        print(f"Model Used:                 {self.embedding_gen.model_name}")
        print(f"Device Used:                {self.embedding_gen.device}")
        print(f"Batch Size:                 {self.embedding_gen.batch_size}")
        
        embeddings_size_mb = self.embeddings_file.stat().st_size / (1024 * 1024)
        metadata_size_mb = self.metadata_file.stat().st_size / (1024 * 1024)
        faiss_size_mb = self.faiss_index_file.stat().st_size / (1024 * 1024)
        
        print(f"Embedding File Size:        {embeddings_size_mb:.2f} MB")
        print(f"Metadata File Size:         {metadata_size_mb:.2f} MB")
        print(f"FAISS Index Size:           {faiss_size_mb:.2f} MB")
        print(f"Total Artifacts Size:       {embeddings_size_mb + metadata_size_mb + faiss_size_mb:.2f} MB")
        
        print(f"Total Runtime:              {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Throughput:                 {self.stats['chunks_processed'] / total_time:.2f} chunks/sec")
        print("=" * 70 + "\n")
        
        logger.info("RAG corpus build complete!")
        logger.info(f"Output artifacts:")
        logger.info(f"  - Embeddings: {self.embeddings_file}")
        logger.info(f"  - Metadata:   {self.metadata_file}")
        logger.info(f"  - FAISS Index: {self.faiss_index_file}")


def main():
    """Entry point."""
    # Additional TensorFlow suppression
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Resolve paths
    script_dir = Path(__file__).parent
    chunks_file = script_dir / 'chunks.jsonl'
    output_dir = script_dir
    
    # Determine batch size based on device availability
    # GPU: batch_size=64 is fine; CPU: use larger batches for efficiency
    has_cuda = torch.cuda.is_available()
    batch_size = 64 if has_cuda else 256  # Larger batches on CPU for 3-4x speedup
    
    logger.info(f"Auto-selected batch_size: {batch_size} (based on device: {'CUDA' if has_cuda else 'CPU'})")
    
    # Build corpus
    builder = RAGCorpusBuilder(
        chunks_file=chunks_file,
        output_dir=output_dir,
        model_name="BAAI/bge-base-en-v1.5",
        batch_size=batch_size,
        device=None  # Auto-detect
    )
    
    builder.build_corpus()


if __name__ == '__main__':
    main()
