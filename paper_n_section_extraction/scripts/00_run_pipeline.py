import fitz
import re
import json
import os
from pathlib import Path

# =============================================
# PIPELINE ORCHESTRATOR - Combines all 3 steps
# =============================================

def get_pdf_path():
    """
    Present user with PDF options to choose from
    """
    print("\n" + "="*60)
    print("PDF SELECTION")
    print("="*60)
    
    # Check what PDFs are available in data folder
    data_folder = "./data"
    if os.path.exists(data_folder):
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    else:
        pdf_files = []
    
    options = pdf_files + ["Enter custom path"]
    
    print("\nAvailable PDFs:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf}")
    if pdf_files:
        print(f"  {len(pdf_files) + 1}. Enter custom path")
    else:
        print("  1. Enter custom path")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect PDF (enter number): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(pdf_files):
                return f"./data/{pdf_files[choice_num - 1]}"
            elif choice_num == len(pdf_files) + 1 or (not pdf_files and choice_num == 1):
                custom_path = input("Enter PDF path (e.g., ./data/chi_2024.pdf): ").strip()
                if os.path.exists(custom_path):
                    return custom_path
                else:
                    print(f"❌ File not found: {custom_path}")
                    continue
            else:
                print("❌ Invalid selection. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")


def step_1_find_paper_starts(pdf_path):
    """
    STEP 1: Find where papers start in the PDF
    """
    print("\n" + "="*60)
    print("STEP 1: Finding Paper Starts")
    print("="*60)
    print(f"📄 PDF: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    paper_starts = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()

        if "ABSTRACT" not in text:
            continue

        lines = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

        title_lines = []

        for line in lines[:20]:
            if (
                "ABSTRACT" in line
                or "CHI '24" in line
                or "@" in line
            ):
                break

            title_lines.append(line)

        if len(title_lines) > 0:
            title = " ".join(title_lines)
            paper_starts.append(
                {
                    "page": page_num,
                    "title": title
                }
            )

    print(f"✅ Paper starts found: {len(paper_starts)}")
    
    # Show first 5 papers
    print("\nFirst 5 papers:")
    for p in paper_starts[:5]:
        print(f"  - Page {p['page']}: {p['title'][:60]}...")

    # Save to file
    os.makedirs("./outputs", exist_ok=True)
    with open(
        "./outputs/paper_start_pages.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            paper_starts,
            f,
            indent=2,
            ensure_ascii=False
        )
    
    print("✅ Saved: ./outputs/paper_start_pages.json\n")
    
    doc.close()
    return paper_starts


def step_2_extract_papers(pdf_path, paper_starts):
    """
    STEP 2: Extract full papers based on start pages
    """
    print("="*60)
    print("STEP 2: Extracting Papers")
    print("="*60)
    print(f"📄 PDF: {pdf_path}")
    print(f"📋 Processing {len(paper_starts)} papers...")
    
    doc = fitz.open(pdf_path)
    papers = []

    for i in range(len(paper_starts)):
        start_page = paper_starts[i]["page"]

        if i < len(paper_starts) - 1:
            end_page = paper_starts[i + 1]["page"] - 1
        else:
            end_page = len(doc) - 1

        full_text = ""

        for page_num in range(start_page, end_page + 1):
            full_text += "\n"
            full_text += doc[page_num].get_text()

        papers.append(
            {
                "title": paper_starts[i]["title"],
                "start_page": start_page,
                "end_page": end_page,
                "text": full_text
            }
        )

    print(f"✅ Total papers extracted: {len(papers)}")

    # Save to file
    with open(
        "./outputs/papers.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            papers,
            f,
            indent=2,
            ensure_ascii=False
        )
    
    print("✅ Saved: ./outputs/papers.json\n")
    
    doc.close()
    return papers


def step_3_extract_sections(papers):
    """
    STEP 3: Extract sections from papers
    """
    print("="*60)
    print("STEP 3: Extracting Sections")
    print("="*60)
    print(f"📋 Processing {len(papers)} papers for sections...")

    SECTION_PATTERN = re.compile(
        r"\n(\d+\s+[A-Z][A-Z\s\-\&\:]{3,})\n"
    )

    structured_papers = []

    for idx, paper in enumerate(papers):
        text = paper["text"]
        matches = list(SECTION_PATTERN.finditer(text))
        sections = {}

        # ABSTRACT
        abs_match = re.search(
            r"ABSTRACT(.*?)(KEYWORDS|1\s+INTRODUCTION)",
            text,
            re.DOTALL
        )

        if abs_match:
            sections["ABSTRACT"] = (
                abs_match.group(1)
                .strip()
            )

        # NUMBERED SECTIONS
        for i in range(len(matches)):
            section_name = (
                matches[i]
                .group(1)
                .replace("\n", " ")
                .strip()
            )

            start = matches[i].end()

            if i < len(matches) - 1:
                end = matches[i + 1].start()
            else:
                end = len(text)

            section_text = text[start:end].strip()

            sections[section_name] = section_text

        structured_papers.append(
            {
                "title": paper["title"],
                "start_page": paper["start_page"],
                "end_page": paper["end_page"],
                "sections": sections
            }
        )
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(papers)} papers...")

    print(f"✅ Total papers with sections: {len(structured_papers)}")

    # Save to file
    with open(
        "./outputs/papers_with_sections.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            structured_papers,
            f,
            indent=2,
            ensure_ascii=False
        )
    
    print("✅ Saved: ./outputs/papers_with_sections.json\n")
    
    return structured_papers


def main():
    """
    Main pipeline orchestrator
    """
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*12 + "PDF PAPER EXTRACTION PIPELINE" + " "*18 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # Step 0: Get PDF path from user
        pdf_path = get_pdf_path()
        print(f"\n✅ Selected PDF: {pdf_path}\n")
        
        # Step 1: Find paper starts
        paper_starts = step_1_find_paper_starts(pdf_path)
        
        # Step 2: Extract papers
        papers = step_2_extract_papers(pdf_path, paper_starts)
        
        # Step 3: Extract sections
        structured_papers = step_3_extract_sections(papers)
        
        # Final summary
        print("="*60)
        print("✅ PIPELINE COMPLETE!")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   • Papers found: {len(structured_papers)}")
        print(f"\n📁 Output files:")
        print(f"   • ./outputs/paper_start_pages.json")
        print(f"   • ./outputs/papers.json")
        print(f"   • ./outputs/papers_with_sections.json\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
