"""
LangGraph node implementations for the agent workflow.
"""
import sys
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import AIMessage, ToolMessage

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings  # noqa: F401 - for future use
from models.state import AgentState
from services.model_service import get_model_with_tools
from graph.prompt import get_system_prompt
from services.document_indexer import DocumentIndexer


async def rag_retrieval_node(state: AgentState) -> Dict:
    """
    RAG retrieval node: searches enabled knowledge bases for relevant context.

    This node runs before the agent node to inject relevant knowledge base
    context into the system prompt.

    Args:
        state: Current agent state

    Returns:
        Updated state with rag_context
    """
    kb_ids = state.get("enabled_knowledge_bases", [])

    if not kb_ids:
        return {"rag_context": ""}

    try:
        indexer = DocumentIndexer()
        last_user_msg = state["messages"][-1].content if state["messages"] else ""

        if not last_user_msg:
            return {"rag_context": ""}

        # Use hybrid retrieval (vector + BM25) with reranking
        context = indexer.get_context_string(
            kb_ids,
            last_user_msg,
            k_per_kb=2,
            use_rerank=True,
            top_n=5,
            use_hybrid=True,  # Enable hybrid retrieval
        )

        return {"rag_context": context}

    except Exception as e:
        # Log error but don't fail the entire flow
        print(f"RAG retrieval error: {e}")
        return {"rag_context": ""}


async def agent_node(state: AgentState) -> Dict:
    """
    Agent node that processes user messages and generates responses.

    This node:
    1. Creates a system message with current context (including RAG)
    2. Calls the LLM with tools bound
    3. Handles tool call detection and confirmation flow

    Args:
        state: Current agent state

    Returns:
        Updated state with AI response and tool call status
    """
    model_with_tools = get_model_with_tools()

    # Get RAG context if available
    rag_context = state.get("rag_context", "")

    system_message = AIMessage(
        content=get_system_prompt(rag_context=rag_context)
    )
    messages = [system_message] + state["messages"]

    response = await model_with_tools.ainvoke(messages)

    has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
    already_confirmed = state.get("tool_call_confirmed", False)

    if has_tool_calls and not already_confirmed:
        # Store pending tool calls and wait for user confirmation
        pending_calls = response.tool_calls
        return {
            "messages": [response],
            "turn_count": state.get("turn_count", 0) + 1,
            "tool_call_confirmed": False,
            "pending_tool_calls": pending_calls,
        }
    elif has_tool_calls and already_confirmed:
        # Already confirmed, proceed normally
        pass
    else:
        pass

    new_turn_count = state.get("turn_count", 0) + 1
    return {
        "messages": [response],
        "turn_count": new_turn_count,
        "tool_call_confirmed": False,
        "pending_tool_calls": None,
    }


async def tool_node(state: AgentState) -> Dict:
    """
    Tool node that executes pending tool calls.

    This node:
    1. Checks if tool calls are confirmed
    2. Executes confirmed tool calls
    3. Returns tool results

    Args:
        state: Current agent state

    Returns:
        State with tool execution results
    """
    from tools.basic_tools import (
        run_command, read_file, write_file, list_directory, find_files,
        get_skill, execute_skill_script,
    )
    from tools.memory_tools import save_memory, load_memory, clear_memory

    # Build base tools dictionary
    tools_by_name = {
        "run_command": run_command,
        "read_file": read_file,
        "write_file": write_file,
        "list_directory": list_directory,
        "find_files": find_files,
        "get_skill": get_skill,
        "execute_skill_script": execute_skill_script,
        "save_memory": save_memory,
        "load_memory": load_memory,
        "clear_memory": clear_memory,
    }

    # Add subagent tools if available
    try:
        from subagent import get_all_subagent_tools
        for tool in get_all_subagent_tools():
            tools_by_name[tool.name] = tool
    except ImportError:
        pass

    last_message = state["messages"][-1]

    # Check if tool calls are confirmed
    if not state.get("tool_call_confirmed", False):
        # Not confirmed yet, don't execute tools
        return {"messages": [], "tool_call_confirmed": False}

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [], "tool_call_confirmed": False}

    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        if tool_name not in tools_by_name:
            result = f"Error: Unknown tool '{tool_name}'"
        else:
            try:
                tool_func = tools_by_name[tool_name]
                result = tool_func.invoke(tool_args)
            except Exception as e:
                result = f"Error: {e}"

        results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {"messages": results, "tool_call_confirmed": False}


def should_continue(state: AgentState) -> str:
    """
    Determine the next node in the graph workflow.

    Args:
        state: Current agent state

    Returns:
        Next node name: "tools", "agent", or "end"
    """
    from config.settings import settings

    if state.get("turn_count", 0) >= settings.MAX_TURNS:
        return "end"

    last_message = state["messages"][-1]
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

    # If has tool calls but not confirmed yet, go to tools node (which will check confirmation)
    if has_tool_calls:
        return "tools"

    if isinstance(last_message, ToolMessage):
        return "agent"

    return "end"
