
import json

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


def create_insight_prompt(context: str) -> str:
    return f"""
You are an AI assistant helping consultants analyse UK government
documents.

Analyse the supplied document context and identify any important:

- risks
- opportunities
- actions
- deadlines
- stakeholders

Only identify information that is supported by the supplied documents.
Do not invent information.

Return ONLY valid JSON using exactly this structure:

{{
    "risks": [],
    "opportunities": [],
    "actions": [],
    "deadlines": [],
    "stakeholders": []
}}

Each item should be a concise statement.

For actions, include the owner if one is explicitly identified.

For deadlines, include the date or timeframe if one is explicitly
identified.

For stakeholders, identify the organisation, group or role mentioned
in the documents.

Document context:

{context}
""".strip()


def generate_insights(prompt: str, llm: LocalLLM) -> dict:
    """Generate structured insights using the already-loaded language model."""
    output = llm.generate(prompt, max_tokens=700)

    start = output.find("{")
    end = output.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("The model did not return a valid JSON object.")

    json_text = output[start:end + 1]

    return json.loads(json_text)


def attach_sources(insights: dict, results: list[dict]) -> dict:
    sources = [
        {
            "filename": result["filename"],
            "chunk": result["chunk_id"],
        }
        for result in results
    ]

    return {
        "insights": insights,
        "retrieved_sources": sources,
    }


def extract_insights(
    retriever: Retriever,
    llm: LocalLLM,
    query: str = "risks opportunities actions deadlines stakeholders",
    top_k: int = 10,
) -> dict:
    """Retrieve relevant document chunks and extract structured insights."""

    results = retriever.search(
        query=query,
        top_k=top_k,
    )

    context = build_context(results)
    prompt = create_insight_prompt(context)

    insights = generate_insights(
        prompt=prompt,
        llm=llm,
    )

    return attach_sources(
        insights=insights,
        results=results,
    )

