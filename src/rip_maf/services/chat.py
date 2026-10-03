def format_context_for_llm(context_json):
    chunks = []
    for i, item in enumerate(context_json.get("results", []), 1):
        chunks.append(
            f"[{i} ({item['source']})]\n" f"{item['section']}\n" f"{item['content']}"
        )
    return "\n\n".join(chunks)

def extract_sources(context_json):
    sources = []
    for item in context_json.get("results", []):
        sources.append(
            {
                "source": item.get("source"),
                "section": item.get("section"),
            }
        )
    unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]

    return unique_sources