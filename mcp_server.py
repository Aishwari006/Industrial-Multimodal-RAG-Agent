from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
import os

load_dotenv()


async def get_exa_tools():
    """
    Connect to the hosted Exa MCP server and return
    all enabled MCP tools.
    """

    client = MultiServerMCPClient(
        {
            "exa": {
                "transport": "streamable_http",
                "url": (
                    "https://mcp.exa.ai/mcp"
                    "?tools=web_search_exa,web_fetch_exa"
                ),
                "headers": {
                    "x-api-key": os.getenv("EXA_API_KEY")
                }
            }
        }
    )

    tools = await client.get_tools()

    return tools