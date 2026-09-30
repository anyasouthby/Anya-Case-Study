from ingestion import load_dataset


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(dataset_path: str) -> list[dict]:
    """Load documents and split them into chunks."""

    documents = load_dataset(dataset_path)

    chunks = []

    for document in documents:
        document_chunks = chunk_text(document["text"])

        for index, chunk in enumerate(document_chunks):
            chunks.append(
                {
                    "filename": document["filename"],
                    "chunk_id": index,
                    "text": chunk,
                }
            )

    return chunks


if __name__ == "__main__":
    chunks = create_chunks("data/dataset")

    print(f"Created {len(chunks)} chunks.")

    for chunk in chunks[:5]:
        print(
            f"\n{chunk['filename']} | "
            f"chunk {chunk['chunk_id']}"
        )
        print(chunk["text"][:200])