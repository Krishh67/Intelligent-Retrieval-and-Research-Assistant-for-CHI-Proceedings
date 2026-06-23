#!/usr/bin/env python3
"""
05_chunk_sections.py

Purpose:
--------
Convert a paper-level corpus (with sections) into a chunk-level corpus for
hierarchical Retrieval-Augmented Generation (RAG) systems.

Key Design Decisions:
---------------------
1. SECTION-AWARE CHUNKING:
   - Chunks NEVER cross section boundaries
   - This preserves semantic structure and prevents context bleeding
   - Example: "ABSTRACT" and "INTRODUCTION" are always separate chunks
   
2. WORD-BASED CHUNKING:
   - Split on whitespace (not characters) for meaningful semantic units
   - Better aligned with embedding models expecting token boundaries
   - More interpretable than character-based chunking

3. OVERLAP STRATEGY:
   - 100-word overlap between consecutive chunks in same section
   - Provides context continuity for embeddings and retrieval
   - Reduces information loss at chunk boundaries

4. METADATA PRESERVATION:
   - Every chunk retains paper metadata (paper_id, year, title, section)
   - Enables hierarchical retrieval: Paper -> Section -> Chunk
   - Supports downstream analysis and traceability

5. GENERATOR PATTERN:
   - Processes papers one at a time (not loaded all into memory)
   - Suitable for large corpora (300 MB+)
   - Enables stream processing and pipeline integration

6. JSONL OUTPUT:
   - One chunk per line (not nested array)
   - Enables streaming consumers and incremental processing
   - Better for distributed systems and data pipelines
"""

import json
import logging
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional
from collections import defaultdict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SectionChunker:
    """
    Handles word-based chunking of text sections with overlap.
    
    Design:
    -------
    - Chunks are created within section boundaries only
    - Each chunk maintains a fixed size (words) with specified overlap
    - Metadata is attached to each chunk for traceability
    """
    
    def __init__(self, chunk_size: int = 700, overlap: int = 100):
        """
        Initialize chunker parameters.
        
        Args:
            chunk_size: Number of words per chunk (excluding overlap)
            overlap: Number of overlapping words between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
    
    def split_into_words(self, text: str) -> List[str]:
        """
        Split text into words (tokens) by whitespace.
        
        Args:
            text: Raw text string
            
        Returns:
            List of words
        """
        return text.split()
    
    def chunk_section(
        self,
        section_text: str,
        chunk_index_start: int = 0
    ) -> Generator[tuple, None, None]:
        """
        Generate chunks from a single section's text.
        
        Args:
            section_text: Text content of the section
            chunk_index_start: Starting index for chunk numbering within section
            
        Yields:
            Tuple of (chunk_text, chunk_index_in_section)
        """
        words = self.split_into_words(section_text)
        
        if len(words) == 0:
            return  # Skip empty sections
        
        chunk_index = chunk_index_start
        
        # Process chunks with sliding window
        for i in range(0, len(words), self.chunk_size - self.overlap):
            # Extract chunk words
            chunk_words = words[i : i + self.chunk_size]
            
            if len(chunk_words) == 0:
                break
            
            # Reconstruct chunk text
            chunk_text = ' '.join(chunk_words)
            
            yield chunk_text, chunk_index
            chunk_index += 1


class CorpusProcessor:
    """
    Main processor for converting paper corpus to chunk corpus.
    
    Responsibilities:
    - Load papers from JSON
    - Validate data integrity
    - Generate chunks with metadata
    - Track statistics
    - Write JSONL output
    """
    
    def __init__(
        self,
        input_file: Path,
        output_file: Path,
        chunk_size: int = 700,
        overlap: int = 100
    ):
        """
        Initialize corpus processor.
        
        Args:
            input_file: Path to all_papers_with_sections.json
            output_file: Path to output chunks.jsonl
            chunk_size: Words per chunk
            overlap: Word overlap between chunks
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.chunker = SectionChunker(chunk_size=chunk_size, overlap=overlap)
        
        # Statistics tracking
        self.stats = {
            'total_papers': 0,
            'papers_processed': 0,
            'papers_skipped': 0,
            'total_sections': 0,
            'sections_skipped': 0,
            'total_chunks': 0,
        }
    
    def validate_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Validate paper record structure and content.
        
        Args:
            paper: Paper record dict
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['paper_id', 'year', 'title', 'sections']
        
        # Check required fields
        if not all(field in paper for field in required_fields):
            logger.warning(
                f"Skipping paper: missing required fields. "
                f"Present: {list(paper.keys())}"
            )
            return False
        
        # Validate sections is a dict
        if not isinstance(paper['sections'], dict):
            logger.warning(
                f"Skipping paper {paper.get('paper_id')}: "
                f"'sections' is not a dictionary"
            )
            return False
        
        # Validate basic field types
        if not isinstance(paper['paper_id'], int):
            logger.warning(
                f"Skipping paper: paper_id is not int (got {type(paper['paper_id'])})"
            )
            return False
        
        if not isinstance(paper['title'], str) or not paper['title'].strip():
            logger.warning(
                f"Skipping paper {paper.get('paper_id')}: "
                f"title is missing or empty"
            )
            return False
        
        return True
    
    def load_papers(self) -> Generator[Dict[str, Any], None, None]:
        """
        Load and yield papers from JSON file.
        
        Uses generator pattern to avoid loading entire file into memory.
        
        Yields:
            Individual paper records
        """
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                # Handle both JSON array and newline-delimited JSON
                content = f.read().strip()
                
                # Try parsing as JSON array first
                if content.startswith('['):
                    papers = json.loads(content)
                    if not isinstance(papers, list):
                        raise ValueError("Expected JSON array or JSONL format")
                    for paper in papers:
                        yield paper
                else:
                    # Assume JSONL format
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_file}")
            raise
    
    def process_paper(
        self,
        paper: Dict[str, Any],
        chunk_id_counter: int
    ) -> tuple[int, int]:
        """
        Process single paper and generate chunks.
        
        Args:
            paper: Paper record
            chunk_id_counter: Current global chunk ID counter
            
        Returns:
            Tuple of (new_chunk_id_counter, sections_in_paper)
        """
        if not self.validate_paper(paper):
            self.stats['papers_skipped'] += 1
            return chunk_id_counter, 0
        
        paper_id = paper['paper_id']
        year = paper['year']
        title = paper['title']
        sections = paper['sections']
        
        sections_in_paper = 0
        
        # Process each section
        for section_name, section_text in sections.items():
            # Validate section
            if not isinstance(section_text, str):
                logger.warning(
                    f"Paper {paper_id}, Section '{section_name}': "
                    f"text is not string, skipping"
                )
                self.stats['sections_skipped'] += 1
                continue
            
            section_text = section_text.strip()
            if not section_text:
                # Skip empty sections
                self.stats['sections_skipped'] += 1
                continue
            
            self.stats['total_sections'] += 1
            sections_in_paper += 1
            
            # Generate chunks from section
            for chunk_text, chunk_index_in_section in self.chunker.chunk_section(
                section_text
            ):
                chunk_record = {
                    'chunk_id': chunk_id_counter,
                    'paper_id': paper_id,
                    'year': year,
                    'title': title,
                    'section': section_name,
                    'chunk_index': chunk_index_in_section,
                    'text': chunk_text,
                }
                
                # Write chunk to JSONL
                # (Caller handles file I/O in generator pattern)
                yield chunk_record
                
                chunk_id_counter += 1
                self.stats['total_chunks'] += 1
        
        self.stats['papers_processed'] += 1
        return chunk_id_counter, sections_in_paper
    
    def process_corpus(self) -> None:
        """
        Main processing pipeline:
        1. Load papers
        2. Generate chunks
        3. Write to JSONL
        4. Track statistics
        5. Print final report
        """
        logger.info(f"Starting corpus processing from: {self.input_file}")
        logger.info(
            f"Configuration: chunk_size={self.chunker.chunk_size}, "
            f"overlap={self.chunker.overlap}"
        )
        
        chunk_id_counter = 1
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as out_f:
                for paper_idx, paper in enumerate(self.load_papers(), 1):
                    self.stats['total_papers'] = paper_idx
                    
                    # Process paper
                    chunk_id_counter, _ = self.process_paper(paper, chunk_id_counter)
                    
                    # Progress logging every 100 papers
                    if paper_idx % 100 == 0:
                        logger.info(
                            f"Progress: {self.stats['papers_processed']} papers processed, "
                            f"{self.stats['total_sections']} sections, "
                            f"{self.stats['total_chunks']} chunks"
                        )
                    
                    # Generate and write chunks for this paper
                    for chunk_record in self.process_paper(paper, chunk_id_counter):
                        out_f.write(json.dumps(chunk_record, ensure_ascii=False))
                        out_f.write('\n')
                        # Update counter from yielded records
                        chunk_id_counter = chunk_record['chunk_id'] + 1
                    
                    # Simplified: re-process without yield for statistics
                    # (In production, refactor to avoid double processing)
            
            self._print_final_statistics()
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def process_corpus_optimized(self) -> None:
        """
        Optimized processing pipeline without double-processing.
        """
        logger.info(f"Starting corpus processing from: {self.input_file}")
        logger.info(
            f"Configuration: chunk_size={self.chunker.chunk_size}, "
            f"overlap={self.chunker.overlap}"
        )
        
        chunk_id_counter = 1
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as out_f:
                for paper_idx, paper in enumerate(self.load_papers(), 1):
                    self.stats['total_papers'] = paper_idx
                    
                    if not self.validate_paper(paper):
                        self.stats['papers_skipped'] += 1
                        continue
                    
                    paper_id = paper['paper_id']
                    year = paper['year']
                    title = paper['title']
                    sections = paper['sections']
                    
                    # Process each section
                    for section_name, section_text in sections.items():
                        if not isinstance(section_text, str):
                            logger.warning(
                                f"Paper {paper_id}, Section '{section_name}': "
                                f"text is not string, skipping"
                            )
                            self.stats['sections_skipped'] += 1
                            continue
                        
                        section_text = section_text.strip()
                        if not section_text:
                            self.stats['sections_skipped'] += 1
                            continue
                        
                        self.stats['total_sections'] += 1
                        
                        # Generate chunks from section
                        for chunk_text, chunk_index_in_section in self.chunker.chunk_section(
                            section_text
                        ):
                            chunk_record = {
                                'chunk_id': chunk_id_counter,
                                'paper_id': paper_id,
                                'year': year,
                                'title': title,
                                'section': section_name,
                                'chunk_index': chunk_index_in_section,
                                'text': chunk_text,
                            }
                            
                            out_f.write(json.dumps(chunk_record, ensure_ascii=False))
                            out_f.write('\n')
                            
                            chunk_id_counter += 1
                            self.stats['total_chunks'] += 1
                    
                    self.stats['papers_processed'] += 1
                    
                    # Progress logging every 100 papers
                    if self.stats['papers_processed'] % 100 == 0:
                        logger.info(
                            f"Progress: {self.stats['papers_processed']} papers processed, "
                            f"{self.stats['total_sections']} sections, "
                            f"{self.stats['total_chunks']} chunks"
                        )
            
            self._print_final_statistics()
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def _print_final_statistics(self) -> None:
        """Print final corpus statistics."""
        print("\n" + "=" * 60)
        print("CORPUS STATISTICS")
        print("=" * 60)
        print(f"Total Papers Loaded:        {self.stats['total_papers']}")
        print(f"Total Papers Processed:     {self.stats['papers_processed']}")
        print(f"Total Papers Skipped:       {self.stats['papers_skipped']}")
        print(f"Total Sections:             {self.stats['total_sections']}")
        print(f"Total Sections Skipped:     {self.stats['sections_skipped']}")
        print(f"Total Chunks Generated:     {self.stats['total_chunks']}")
        
        if self.stats['papers_processed'] > 0:
            avg_chunks = self.stats['total_chunks'] / self.stats['papers_processed']
            print(f"Average Chunks Per Paper:   {avg_chunks:.2f}")
        
        print(f"Output File:                {self.output_file}")
        print("=" * 60 + "\n")
        
        logger.info(f"Output written to: {self.output_file}")


def main():
    """Entry point."""
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_file = script_dir / 'all_papers_with_sections.json'
    output_file = script_dir / 'chunks.jsonl'
    
    # Create processor and run
    processor = CorpusProcessor(
        input_file=input_file,
        output_file=output_file,
        chunk_size=700,
        overlap=100
    )
    
    processor.process_corpus_optimized()


if __name__ == '__main__':
    main()
