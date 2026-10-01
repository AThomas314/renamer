from pathlib import Path
import re
import pymupdf
import requests


# Directory containing your PDFs
TARGET_DIR = Path("/home/ashish/Downloads/INDS/")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:7b"


def extract_first_page_text(pdf_path: Path, max_chars: int = 2000) -> str:
    """Extracts text from the first page of a PDF up to max_chars."""
    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            return ""
        first_page = doc[0]
        text = first_page.get_text("text")
        return text[:max_chars].strip()
    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")
        return ""


def ask_qwen_for_title(text: str) -> str:
    """Sends the first-page text to qwen2.5-coder to identify the exact paper title."""
    prompt = f"""You are an expert academic paper processor.
Extract the EXACT main title of the research paper from the text below.

Rules:
1. Output ONLY the title. No preamble, no quotes, no explanations.
2. Exclude journal names, volume/issue numbers, DOIs, publisher names (e.g. Elsevier, Springer, IEEE, MDPI, Applied Sciences, Sensors, Polymers), author names, and page numbers.
3. Keep the title clean and properly capitalized.

Text from First Page:
---
{text}
---

Title:"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Keep output deterministic
            "num_predict": 128,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if resp.status_code == 200:
            title = resp.json().get("response", "").strip()
            # Clean up potential LLM output artifacts
            title = re.sub(r'^["\']|["\']$', "", title)
            title = title.replace("\n", " ")
            return title
    except Exception as e:
        print(f"Ollama API request failed: {e}")
    return ""


def sanitize_filename(name: str, max_length: int = 150) -> str:
    """Sanitizes paper title into a clean, filesystem-safe filename."""
    # Replace illegal filename characters with underscores/hyphens
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:max_length]


def process_pdf_renaming(dry_run: bool = True):
    """Processes all PDFs in TARGET_DIR and renames them based on extracted titles."""
    pdf_files = list(TARGET_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {TARGET_DIR.resolve()}\n")

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")
        page_text = extract_first_page_text(pdf)

        if not page_text:
            print(f"  └─ Skipping: Could not extract text from first page.")
            continue

        extracted_title = ask_qwen_for_title(page_text)

        if not extracted_title:
            print(f"  └─ Skipping: Model failed to extract title.")
            continue

        safe_title = sanitize_filename(extracted_title)
        new_filename = f"{safe_title}.pdf"
        new_path = pdf.parent / new_filename

        if dry_run:
            print(f"  └─ [DRY RUN] Would rename to: {new_filename}")
        else:
            if new_path.exists() and new_path != pdf:
                print(f"  └─ Target file {new_filename} already exists. Skipping.")
                continue
            pdf.rename(new_path)
            print(f"  └─ Renamed to: {new_filename}")


if __name__ == "__main__":
    # Set dry_run=False to execute real file renaming on disk
    process_pdf_renaming(dry_run=False)
