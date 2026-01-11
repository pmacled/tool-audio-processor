#!/usr/bin/env python3
"""
MCP Server for Audio Layer Manipulation
Provides tools for separating, analyzing, synthesizing, and manipulating audio layers.
"""

from fastmcp import FastMCP
from tools import register_all_tools

# Initialize FastMCP server
mcp = FastMCP("audio-processor")

# Register all tools
register_all_tools(mcp)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
