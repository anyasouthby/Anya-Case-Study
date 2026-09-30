
from retrieval import Retriever
from llm import LocalLLM
from qa import answer_question
from insights import extract_insights
from summary import generate_executive_summary


def route_request(request: str, llm: LocalLLM) -> str:
    """
    Decide which capability should handle the user's request.

    High-confidence intents use deterministic rules.
    The language model is used as a fallback for ambiguous requests.
    """

    request_lower = request.lower()

    # High-confidence summary requests
    summary_keywords = [
        "summary",
        "summarise",
        "summarize",
        "executive summary",
    ]

    if any(keyword in request_lower for keyword in summary_keywords):
        return "summary"

    # High-confidence insight requests
    insight_keywords = [
        "risk",
        "risks",
        "opportunit",
        "action",
        "actions",
        "deadline",
        "deadlines",
        "stakeholder",
        "stakeholders",
    ]

    if any(keyword in request_lower for keyword in insight_keywords):
        return "insights"

    # Use the language model as a fallback for ambiguous requests.
    routing_prompt = f"""
You are routing a user's request to one of four capabilities.

Choose exactly one of:

- qa
- insights
- summary

Use:
- qa for factual questions about the documents
- insights for requests about risks, opportunities, actions,
  deadlines or stakeholders
- summary for requests asking for a summary

Return ONLY the capability name.

User request:
{request}
""".strip()

    result = llm.generate(
        routing_prompt,
        max_tokens=20,
    ).lower().strip()

    if "summary" in result:
        return "summary"

    if "insights" in result:
        return "insights"

    return "qa"


def run_agent(
    request: str,
    retriever: Retriever,
    llm: LocalLLM,
) -> dict:
    """
    Route a user request to the appropriate document-analysis capability.
    """

    tool = route_request(
        request=request,
        llm=llm,
    )

    if tool == "qa":
        answer, sources = answer_question(
            question=request,
            retriever=retriever,
            llm=llm,
        )

        return {
            "tool": "qa",
            "answer": answer,
            "sources": sources,
        }

    if tool == "insights":
        result = extract_insights(
            retriever=retriever,
            llm=llm,
        )

        return {
            "tool": "insights",
            **result,
        }

    if tool == "summary":
        summary, sources = generate_executive_summary(
            retriever=retriever,
            llm = llm,
        )

        return {
            "tool": "summary",
            "summary": summary,
            "sources": sources,
        }

    raise ValueError(f"Unknown tool: {tool}")
