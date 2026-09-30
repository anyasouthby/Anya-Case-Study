
from retrieval import Retriever
from llm import LocalLLM


MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def build_context(results: list[dict]) -> str:
    """Build a context string from retrieved document chunks."""

    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['filename']}\n"
            f"Chunk: {result['chunk_id']}\n"
            f"{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def create_prompt(question: str, context: str) -> str:
    """Create a grounded prompt for the LLM."""

    return f"""
You are an AI assistant helping consultants analyse a collection
of UK government documents.

Answer the user's question using ONLY the supplied document context.

If the supplied context does not contain enough information to answer
the question, say that the retrieved context does not provide enough
information.

Do not claim that a source document contains no such information unless
the supplied context explicitly supports that claim.

For every substantive claim in your answer, identify the source
document that supports it using the filename provided in the context.

Do not invent sources or cite documents that were not provided.

Keep your answer concise and factual.

User question:
{question}

Document context:
{context}
""".strip()


def generate_answer(prompt: str, llm: LocalLLM) -> str:
    return llm.generate(prompt, max_tokens=500)


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LocalLLM,
    top_k: int = 5,
) -> tuple[str, list[dict]]:
    """Retrieve relevant context and generate an answer."""

    results = retriever.search(
        query=question,
        top_k=top_k,
    )

    context = build_context(results)

    prompt = create_prompt(
        question=question,
        context=context,
    )

    answer = generate_answer(prompt, llm)

    return answer, results


if __name__ == "__main__":
    retriever = Retriever("data/dataset")
    llm = LocalLLM()

    question = "What are the main risks associated with AI?"

    answer, sources = answer_question(
    question=question,
    retriever=retriever,
    llm=llm,
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in sources:
        print(
            f"- {source['filename']} "
            f"(chunk {source['chunk_id']})"
        )

