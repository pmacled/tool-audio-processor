"""
Audio processing tools for the MCP server.
"""

from .separation import register_tools as register_separation_tools
from .analysis import register_tools as register_analysis_tools
from .synthesis import register_tools as register_synthesis_tools
from .mixing import register_tools as register_mixing_tools
from .effects import register_tools as register_effects_tools
from .midi_extraction import register_tools as register_midi_extraction_tools
from .midi_refinement import register_tools as register_midi_refinement_tools
from .notation import register_tools as register_notation_tools


def register_all_tools(mcp):
    """Register all tools with the MCP server."""
    register_separation_tools(mcp)
    register_analysis_tools(mcp)
    register_synthesis_tools(mcp)
    register_mixing_tools(mcp)
    register_effects_tools(mcp)
    register_midi_extraction_tools(mcp)
    register_midi_refinement_tools(mcp)
    register_notation_tools(mcp)


__all__ = [
    'register_all_tools',
    'register_separation_tools',
    'register_analysis_tools',
    'register_synthesis_tools',
    'register_mixing_tools',
    'register_effects_tools',
    'register_midi_extraction_tools',
    'register_midi_refinement_tools',
    'register_notation_tools',
]
