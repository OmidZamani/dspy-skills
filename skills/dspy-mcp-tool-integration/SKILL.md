---
name: dspy-mcp-tool-integration
version: "1.0.0"
dspy-compatibility: "3.2.1"
tags: ["agent", "production"]
requires-extras: ["dspy[mcp]"]
description: Use for MCP tools with DSPy, Model Context Protocol servers, dspy.Tool.from_mcp_tool, and ReAct agents over MCP-compatible tools.
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
---

# DSPy MCP Tool Integration

## Goal

Connect an MCP server with the MCP Python client, convert its tools to `dspy.Tool`, and use them in an async DSPy agent.

## Install

```bash
pip install -U "dspy[mcp]>=3.2.1,<3.3"
```

DSPy converts tools but does not manage MCP connections. Keep the `ClientSession` alive for as long as the DSPy tools are in use.

## Streamable HTTP Server

```python
import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            tools = [dspy.Tool.from_mcp_tool(session, tool) for tool in response.tools]

            agent = dspy.ReAct("task -> result", tools=tools, max_iters=5)
            output = await agent.acall(task="Check the weather in Tokyo")
            print(output.result)

asyncio.run(main())
```

## Local Stdio Server

```python
import asyncio
import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="python3",
        args=["path/to/server.py"],
        env=None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            tools = [dspy.Tool.from_mcp_tool(session, tool) for tool in response.tools]
            agent = dspy.ReAct("question -> answer", tools=tools, max_iters=5)
            print((await agent.acall(question="What is 25 + 17?")).answer)

asyncio.run(main())
```

## Best Practices

1. Use `acall()` because MCP tools are asynchronous.
2. Initialize the session before listing tools.
3. Keep tool descriptions precise at the MCP server boundary.
4. Apply authentication and authorization before exposing sensitive tools.
5. Set a modest `max_iters` and trace tool use in production.

## Related Skills

- Build agents: [dspy-react-agent-builder](../dspy-react-agent-builder/SKILL.md)
- Configure native function calling: [dspy-adapters-multimodal](../dspy-adapters-multimodal/SKILL.md)
- Add async runtime patterns: [dspy-production-deployment](../dspy-production-deployment/SKILL.md)

## Official Documentation

- **DSPy MCP guide**: https://dspy.ai/learn/programming/mcp/
- **DSPy MCP tutorial**: https://dspy.ai/tutorials/mcp/
- **MCP Python SDK**: https://github.com/modelcontextprotocol/python-sdk
