
from retrieval import Retriever
from llm import LocalLLM


def build_context(results: list[dict]) -> str:
    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['filename']}\n"
            f"Chunk: {result['chunk_id']}\n"
            f"{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def create_summary_prompt(context: str) -> str:
    return f"""
You are an AI assistant helping consultants analyse UK government
documents.

Create a concise executive summary of the most important information
in the supplied document context.

Focus on:

- key strategic points
- important risks
- opportunities
- actions or implications

Only include information supported by the supplied documents.
Do not invent information.

The summary must be no more than 300 words.

Do not include a Sources section. Sources will be displayed separately.

Document context:

{context}
""".strip()


def limit_to_300_words(text: str) -> str:
    """Ensure the final summary contains no more than 300 words."""

    words = text.split()

    if len(words) <= 300:
        return text.strip()

    truncated = " ".join(words[:300])

    # Prefer ending at the last complete sentence.
    last_sentence = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )

    if last_sentence > 0:
        return truncated[:last_sentence + 1].strip()

    return truncated.strip()


def generate_summary(
    prompt: str,
    llm: LocalLLM,
) -> str:
    """Generate an executive summary using the loaded language model."""

    output = llm.generate(
        prompt,
        max_tokens=500,
    )

    return limit_to_300_words(output)


def generate_executive_summary(
    retriever: Retriever,
    llm: LocalLLM,
    top_k: int = 10,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant document chunks and generate an executive summary.
    """

    results = retriever.search(
        query="key strategic points risks opportunities actions",
        top_k=top_k,
    )

    context = build_context(results)

    prompt = create_summary_prompt(context)

    summary = generate_summary(
        prompt=prompt,
        llm=llm,
    )

    return summary, results
