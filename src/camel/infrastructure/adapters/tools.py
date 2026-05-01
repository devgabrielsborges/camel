from __future__ import annotations

from agents import FunctionTool, RunContextWrapper, function_tool

from camel.domain.value_objects import Chunk


def build_knowledge_tool(chunks: list[Chunk]) -> FunctionTool:
    """Build a search_knowledge_base tool scoped to a specific record's chunks."""
    content_blocks = [chunk.content for chunk in chunks]
    joined = "\n\n---\n\n".join(content_blocks) if content_blocks else ""

    async def search_knowledge_base(ctx: RunContextWrapper[None], query: str) -> str:
        """Search the knowledge base for reference material related to the user query.

        Args:
            query: The search query describing what information is needed.
        """
        if not joined:
            return "No knowledge base content available for this topic."
        return joined

    return function_tool(
        search_knowledge_base,
        name_override="search_knowledge_base",
        description_override=(
            "Search the knowledge base for reference material. "
            "Call this tool before answering the user's question."
        ),
    )
