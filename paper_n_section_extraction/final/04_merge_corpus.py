"""
Merge Multiple Paper JSON Files with Metadata

This script merges multiple papers_with_sections.json files into a single
unified corpus with globally unique paper IDs and year metadata.

Input files:
- 21.1_papers_with_sections.json
- 21.2_papers_with_sections.json
- 21.3_papers_with_sections.json
- 23_papers_with_sections.json
- 24_papers_with_sections.json

Output:
- all_papers_with_sections.json (merged corpus with metadata)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def extract_year_from_filename(filename: str) -> int:
    """
    Extract year from filename.
    
    Examples:
        "21.1_papers_with_sections.json" -> 2021
        "23_papers_with_sections.json" -> 2023
        "24_papers_with_sections.json" -> 2024
    
    Args:
        filename: Name of the JSON file
        
    Returns:
        Year as 4-digit integer (20xx format)
    """
    # Extract the prefix before underscore (e.g., "21.1", "23", "24")
    prefix = filename.split("_")[0]
    
    # Handle formats like "21.1" or "23" or "24"
    year_str = prefix.split(".")[0]  # Get "21", "23", "24"
    
    # Convert to 4-digit year
    year_2digit = int(year_str)
    year_4digit = 2000 + year_2digit
    
    return year_4digit


def load_json_file(file_path: Path) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Load and validate JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Tuple of (success: bool, data: list or [], error_message: str)
    """
    try:
        # Check if file exists
        if not file_path.exists():
            return False, [], f"File not found: {file_path}"
        
        # Load JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate it's a list
        if not isinstance(data, list):
            return False, [], f"Expected list, got {type(data).__name__}"
        
        return True, data, ""
        
    except json.JSONDecodeError as e:
        return False, [], f"Invalid JSON: {e}"
    except Exception as e:
        return False, [], f"Error reading file: {e}"


def validate_paper_entry(paper: Dict[str, Any], idx: int) -> Tuple[bool, str]:
    """
    Validate a single paper entry has required fields.
    
    Args:
        paper: Paper dictionary
        idx: Index in the list (for error messages)
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    required_fields = ["title", "start_page", "end_page", "sections"]
    
    if not isinstance(paper, dict):
        return False, f"Entry {idx}: Expected dict, got {type(paper).__name__}"
    
    missing = [f for f in required_fields if f not in paper]
    if missing:
        return False, f"Entry {idx}: Missing fields {missing}"
    
    return True, ""


def merge_papers(
    file_list: List[Path],
    final_dir: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Load all JSON files, merge them with metadata, and assign unique IDs.
    
    Args:
        file_list: List of filenames to load
        final_dir: Directory containing the files
        
    Returns:
        Tuple of (merged_papers: list, stats: dict)
    """
    merged_papers = []
    stats = {}
    paper_id_counter = 0
    
    print("\n" + "="*70)
    print("LOADING AND MERGING PAPERS")
    print("="*70)
    
    # Process each file
    for filename in file_list:
        file_path = final_dir / filename
        
        print(f"\n📂 Processing: {filename}")
        
        # Extract year from filename
        year = extract_year_from_filename(filename)
        print(f"   Year: {year}")
        
        # Load JSON file
        success, papers, error_msg = load_json_file(file_path)
        
        if not success:
            print(f"   ❌ Error: {error_msg}")
            stats[filename] = {"loaded": 0, "skipped": 0, "year": year}
            continue
        
        print(f"   ✅ Loaded: {len(papers)} papers")
        
        # Process each paper in this file
        papers_added = 0
        papers_skipped = 0
        
        for idx, paper in enumerate(papers):
            # Validate entry
            is_valid, error_msg = validate_paper_entry(paper, idx)
            
            if not is_valid:
                print(f"      ⚠️  {error_msg} - SKIPPING")
                papers_skipped += 1
                continue
            
            # Assign unique paper_id and year
            paper_id_counter += 1
            
            # Create enriched paper entry with metadata
            enriched_paper = {
                "paper_id": paper_id_counter,
                "year": year,
                "title": paper["title"],
                "start_page": paper["start_page"],
                "end_page": paper["end_page"],
                "sections": paper["sections"]
            }
            
            merged_papers.append(enriched_paper)
            papers_added += 1
        
        # Store statistics for this file
        stats[filename] = {
            "loaded": papers_added,
            "skipped": papers_skipped,
            "year": year,
            "total": len(papers)
        }
        
        print(f"   📊 Added: {papers_added}, Skipped: {papers_skipped}")
    
    return merged_papers, stats


def save_merged_corpus(
    papers: List[Dict[str, Any]],
    output_path: Path
) -> bool:
    """
    Save merged papers to output JSON file.
    
    Args:
        papers: List of enriched paper dictionaries
        output_path: Path where to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False


def print_statistics(papers: List[Dict[str, Any]], stats: Dict[str, Any]):
    """
    Print summary statistics and sample papers.
    
    Args:
        papers: List of merged papers
        stats: Statistics dictionary from merge process
    """
    print("\n" + "="*70)
    print("MERGE STATISTICS")
    print("="*70)
    
    # Per-file statistics
    print("\n📋 Papers per file:")
    total_loaded = 0
    total_skipped = 0
    
    for filename in sorted(stats.keys()):
        info = stats[filename]
        print(
            f"   {filename:40s} | "
            f"Loaded: {info['loaded']:3d} | "
            f"Skipped: {info['skipped']:2d} | "
            f"Year: {info['year']}"
        )
        total_loaded += info["loaded"]
        total_skipped += info["skipped"]
    
    # Overall statistics
    print("\n📊 Overall Statistics:")
    print(f"   Total papers merged: {len(papers)}")
    print(f"   Total papers loaded: {total_loaded}")
    print(f"   Total papers skipped: {total_skipped}")
    
    # Year breakdown
    year_count = {}
    for paper in papers:
        year = paper["year"]
        year_count[year] = year_count.get(year, 0) + 1
    
    print("\n📅 Papers by year:")
    for year in sorted(year_count.keys()):
        print(f"   {year}: {year_count[year]} papers")
    
    # Sample papers
    if papers:
        print("\n📄 First paper metadata:")
        first = papers[0]
        print(f"   ID: {first['paper_id']}")
        print(f"   Year: {first['year']}")
        print(f"   Title: {first['title'][:70]}...")
        print(f"   Pages: {first['start_page']}-{first['end_page']}")
        print(f"   Sections: {', '.join(list(first['sections'].keys())[:3])}...")
        
        print("\n📄 Last paper metadata:")
        last = papers[-1]
        print(f"   ID: {last['paper_id']}")
        print(f"   Year: {last['year']}")
        print(f"   Title: {last['title'][:70]}...")
        print(f"   Pages: {last['start_page']}-{last['end_page']}")
        print(f"   Sections: {', '.join(list(last['sections'].keys())[:3])}...")


def main():
    """
    Main orchestrator function.
    """
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*14 + "PAPER CORPUS MERGE - METADATA ENRICHMENT" + " "*14 + "║")
    print("╚" + "="*68 + "╝")
    
    # Define file list
    files_to_merge = [
        "21.1_papers_with_sections.json",
        "21.2_papers_with_sections.json",
        "21.3_papers_with_sections.json",
        "23_papers_with_sections.json",
        "24_papers_with_sections.json",
    ]
    
    # Use pathlib for directory resolution
    script_dir = Path(__file__).parent
    final_dir = script_dir.parent / "final"
    output_path = script_dir.parent / "outputs" / "all_papers_with_sections.json"
    
    print(f"\n📁 Working directories:")
    print(f"   Source: {final_dir}")
    print(f"   Output: {output_path.parent}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge papers
    merged_papers, stats = merge_papers(files_to_merge, final_dir)
    
    # Print statistics
    print_statistics(merged_papers, stats)
    
    # Save merged corpus
    print("\n" + "="*70)
    print("SAVING MERGED CORPUS")
    print("="*70)
    
    success = save_merged_corpus(merged_papers, output_path)
    
    if success:
        print(f"\n✅ Successfully saved: {output_path}")
        print(f"   Total papers in corpus: {len(merged_papers)}")
        print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"\n❌ Failed to save corpus")
        return 1
    
    print("\n" + "="*70)
    print("✅ MERGE COMPLETE")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
