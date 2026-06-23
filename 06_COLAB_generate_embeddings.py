# Google Colab - Complete Code Ready to Paste (All-in-One)

## Instructions:
# 1. In Colab Cell 1: Run the installation
# 2. In Colab Cell 2: Set up paths
# 3. In Colab Cell 3: Paste ENTIRE code below
# 4. In Colab Cell 4: Run main()

# ============================================================================
# COLAB CELL 1: INSTALL DEPENDENCIES
# ============================================================================
"""
!pip install -q sentence-transformers torch faiss-gpu numpy
"""

# ============================================================================
# COLAB CELL 2: SETUP PATHS
# ============================================================================
"""
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# UPDATE THIS PATH WITH YOUR ACTUAL chunks.jsonl LOCATION
file_path = '/content/drive/MyDrive/RAG_DATA/chunks.jsonl'

# Verify file exists
import os
if os.path.exists(file_path):
    print(f"✓ Found chunks.jsonl: {file_path}")
else:
    print(f"✗ Not found: {file_path}")
"""

# ============================================================================
# COLAB CELL 3: PASTE CODE BELOW (EVERYTHING AFTER THIS LINE)
# ============================================================================

import os
import json
import logging
import time
from pathlib import Path
from typing import Generator, List, Dict, Any, Tuple, Optional
import pickle
import sys

os.environ['TF_CPP_LOGGING_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss

IN_COLAB = 'google.colab' in sys.modules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_colab_environment():
    if IN_COLAB:
        from google.colab import drive
        logger.info("Setting up Google Colab environment...")
        output_dir = Path('/content/drive/MyDrive/rag_embeddings')
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        return output_dir
    else:
        return Path.cwd()

class EmbeddingGenerator:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 128, device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def prepare_text_for_embedding(self, title: str, section: str, text: str) -> str:
        combined_text = f"Title: {title}\n\nSection: {section}\n\n{text}"
        return combined_text
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

class ChunkDataLoader:
    def __init__(self, jsonl_file: Path):
        self.jsonl_file = Path(jsonl_file)
    
    def load_chunks(self) -> Generator[Dict[str, Any], None, None]:
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
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.jsonl_file}")
            raise
    
    def validate_chunk(self, chunk: Dict[str, Any]) -> Tuple[bool, str]:
        required_fields = ['chunk_id', 'paper_id', 'year', 'title', 'section', 'text']
        missing = [f for f in required_fields if f not in chunk]
        if missing:
            return False, f"Missing fields: {missing}"
        if not isinstance(chunk['text'], str) or not chunk['text'].strip():
            return False, "Text is empty or not a string"
        return True, ""

class RAGCorpusBuilder:
    def __init__(self, chunks_file: Path, output_dir: Path, model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 128, device: str = None, checkpoint_interval: int = 10000):
        self.chunks_file = Path(chunks_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = self.output_dir / 'embeddings.npy'
        self.metadata_file = self.output_dir / 'metadata.pkl'
        self.faiss_index_file = self.output_dir / 'faiss.index'
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_interval = checkpoint_interval
        self.embedding_gen = EmbeddingGenerator(model_name=model_name, batch_size=batch_size, device=device)
        self.data_loader = ChunkDataLoader(self.chunks_file)
        
        self.stats = {'total_chunks': 0, 'chunks_processed': 0, 'chunks_skipped': 0, 'start_time': None, 'end_time': None}
        self.all_embeddings = []
        self.all_metadata = []
    
    def build_corpus(self) -> None:
        logger.info("=" * 70)
        logger.info("RAG CORPUS BUILDER (COLAB OPTIMIZED)")
        logger.info("=" * 70)
        logger.info(f"Input file: {self.chunks_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model: {self.embedding_gen.model_name}")
        logger.info(f"Batch size: {self.embedding_gen.batch_size}")
        logger.info(f"Device: {self.embedding_gen.device}")
        logger.info("=" * 70)
        
        self.stats['start_time'] = time.time()
        logger.info("\n[Phase 1] Generating embeddings with checkpointing...")
        self._generate_embeddings()
        
        logger.info("\n[Phase 2] Building FAISS index...")
        self._build_faiss_index()
        
        logger.info("\n[Phase 3] Saving final artifacts...")
        self._save_embeddings()
        self._save_metadata()
        self._save_faiss_index()
        
        self.stats['end_time'] = time.time()
        logger.info("\n[Phase 4] Printing statistics...")
        self._print_statistics()
    
    def _save_checkpoint(self, checkpoint_num: int) -> None:
        try:
            checkpoint_file = self.checkpoint_dir / f'checkpoint_{checkpoint_num}.pkl'
            checkpoint_data = {
                'chunks_processed': self.stats['chunks_processed'],
                'embeddings': np.vstack(self.all_embeddings) if self.all_embeddings else None,
                'metadata': self.all_metadata,
                'timestamp': time.time()
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _generate_embeddings(self) -> None:
        batch_texts = []
        batch_metadata = []
        batch_start_time = time.time()
        
        for chunk in self.data_loader.load_chunks():
            self.stats['total_chunks'] += 1
            is_valid, reason = self.data_loader.validate_chunk(chunk)
            if not is_valid:
                logger.warning(f"Skipping chunk {chunk.get('chunk_id', '?')}: {reason}")
                self.stats['chunks_skipped'] += 1
                continue
            
            text_for_embedding = self.embedding_gen.prepare_text_for_embedding(
                title=chunk['title'], section=chunk['section'], text=chunk['text']
            )
            
            batch_texts.append(text_for_embedding)
            batch_metadata.append({
                'chunk_id': chunk['chunk_id'],
                'paper_id': chunk['paper_id'],
                'year': chunk['year'],
                'title': chunk['title'],
                'section': chunk['section'],
                'chunk_index': chunk['chunk_index'],
            })
            
            if len(batch_texts) >= self.embedding_gen.batch_size:
                embeddings = self.embedding_gen.generate_embeddings_batch(batch_texts)
                self.all_embeddings.append(embeddings)
                self.all_metadata.extend(batch_metadata)
                self.stats['chunks_processed'] += len(batch_texts)
                batch_texts = []
                batch_metadata = []
                
                if self.stats['chunks_processed'] % 1000 == 0:
                    elapsed = time.time() - batch_start_time
                    chunks_per_sec = self.stats['chunks_processed'] / elapsed
                    total_elapsed = time.time() - self.stats['start_time']
                    remaining = ((self.stats['total_chunks'] - self.stats['chunks_processed']) / chunks_per_sec) if self.stats['chunks_processed'] > 0 else 0
                    logger.info(f"Progress: {self.stats['chunks_processed']} chunks | {chunks_per_sec:.1f} ch/sec | Elapsed: {total_elapsed:.1f}s | Remaining: {remaining:.1f}s")
                
                if self.stats['chunks_processed'] % self.checkpoint_interval == 0:
                    checkpoint_num = self.stats['chunks_processed'] // self.checkpoint_interval
                    self._save_checkpoint(checkpoint_num)
        
        if batch_texts:
            embeddings = self.embedding_gen.generate_embeddings_batch(batch_texts)
            self.all_embeddings.append(embeddings)
            self.all_metadata.extend(batch_metadata)
            self.stats['chunks_processed'] += len(batch_texts)
        
        self.embeddings_array = np.vstack(self.all_embeddings)
        logger.info(f"Embeddings generated: {self.embeddings_array.shape}")
    
    def _build_faiss_index(self) -> None:
        logger.info("Normalizing embeddings...")
        self.embeddings_array_normalized = self.embeddings_array.copy().astype(np.float32)
        faiss.normalize_L2(self.embeddings_array_normalized)
        
        logger.info("Creating FAISS IndexFlatIP...")
        self.index = faiss.IndexFlatIP(self.embeddings_array_normalized.shape[1])
        self.index.add(self.embeddings_array_normalized)
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
    
    def _save_embeddings(self) -> None:
        np.save(self.embeddings_file, self.embeddings_array_normalized)
        file_size_mb = self.embeddings_file.stat().st_size / (1024 * 1024)
        logger.info(f"Embeddings saved: {self.embeddings_file} ({file_size_mb:.2f} MB)")
    
    def _save_metadata(self) -> None:
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.all_metadata, f)
        file_size_mb = self.metadata_file.stat().st_size / (1024 * 1024)
        logger.info(f"Metadata saved: {self.metadata_file} ({file_size_mb:.2f} MB)")
    
    def _save_faiss_index(self) -> None:
        faiss.write_index(self.index, str(self.faiss_index_file))
        file_size_mb = self.faiss_index_file.stat().st_size / (1024 * 1024)
        logger.info(f"FAISS index saved: {self.faiss_index_file} ({file_size_mb:.2f} MB)")
    
    def _print_statistics(self) -> None:
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

def main():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    output_dir = setup_colab_environment() if IN_COLAB else Path.cwd()
    
    # Get file_path from Colab cell (defined in Cell 2)
    chunks_file = Path(file_path)  # noqa
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    builder = RAGCorpusBuilder(
        chunks_file=chunks_file,
        output_dir=output_dir,
        model_name="BAAI/bge-large-en-v1.5",
        batch_size=128,
        device=None,
        checkpoint_interval=10000
    )
    
    builder.build_corpus()

# ============================================================================
# COLAB CELL 4: RUN THIS
# ============================================================================
"""
main()
"""
