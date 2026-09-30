from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


def is_pdf(file_path: Path) -> bool:
    """Check whether a file is actually a PDF."""
    with open(file_path, "rb") as file:
        header = file.read(5)

    return header == b"%PDF-"


def extract_pdf_text(file_path: Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_html_text(file_path: Path) -> str:
    """Extract readable content from an HTML document."""

    html = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that are not useful for document analysis.
    for element in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
    ]):
        element.decompose()

    # Remove common GOV.UK page elements by their text.
    unwanted_phrases = [
        "Print this page",
        "Get emails about this page",
        "Cookies on GOV.UK",
        "Accept all cookies",
        "Reject optional cookies",
        "Set cookie preferences",
        "You have accepted additional cookies.",
        "You have rejected additional cookies.",
    ]

    for element in soup.find_all(string=True):
        if element.strip() in unwanted_phrases:
            element.parent.decompose()

    main_content = soup.find("main")

    if main_content:
        return main_content.get_text(
            separator="\n",
            strip=True,
        )

    return soup.get_text(
        separator="\n",
        strip=True,
    )

def load_document(file_path: Path) -> str:
    """Load a document and determine its actual format."""

    if is_pdf(file_path):
        return extract_pdf_text(file_path)

    return extract_html_text(file_path)


def load_dataset(dataset_path: str) -> list[dict]:
    """Load all documents from a dataset directory."""

    dataset = []

    for file_path in sorted(Path(dataset_path).iterdir()):

        if not file_path.is_file():
            continue

        if file_path.name.startswith("."):
            continue

        text = load_document(file_path)

        dataset.append(
            {
                "filename": file_path.name,
                "text": text,
            }
        )

    return dataset


if __name__ == "__main__":
    documents = load_dataset("data/dataset")

    print(f"Loaded {len(documents)} documents.")

    for document in documents:
        print(
            f"{document['filename']}: "
            f"{len(document['text']):,} characters"
        )