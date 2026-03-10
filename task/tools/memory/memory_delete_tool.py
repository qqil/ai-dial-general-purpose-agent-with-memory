from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class DeleteMemoryTool(BaseTool):
    """
    Tool for deleting all long-term memories about the user.

    This permanently removes all stored memories from the system.
    Use with caution - this action cannot be undone.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        # TODO: provide self-descriptive name
        return "delete_memory"

    @property
    def description(self) -> str:
        # TODO: provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return "This tool is used to permanently delete all long-term memories about the user from the system. " \
            "Use this tool with caution, as it will remove all stored information about the user and cannot be undone. " \
            "This tool should only be used if you want to completely reset the memory store and remove all previously stored facts about the user. " \
            "After using this tool, the system will no longer have access to any past information about the user, so it should only be used in situations where you want to start fresh without any prior context."

    @property
    def parameters(self) -> dict[str, Any]:
        # TODO: provide tool parameters JSON Schema with empty properties
        return {
            "type": "object",
            "properties": {}
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        #TODO:
        # 1. Call `memory_store` `delete_all_memories` (we will implement logic in `memory_store` later
        # 2. Add result to stage
        # 3. Return result
        api_key = tool_call_params.api_key

        result = await self.memory_store.delete_all_memories(api_key=api_key)
        tool_call_params.stage.append_content(f"delete_memory_result: {result}")

        return result