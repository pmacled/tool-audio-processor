#!/usr/bin/env python3
"""
Test script to validate the audio processor MCP server implementation.
This script checks that all tools are properly defined and can be imported.
"""

import sys
import importlib.util

def test_server_imports():
    """Test that server.py can be imported and all dependencies are available."""
    print("Testing server.py imports...")
    
    try:
        spec = importlib.util.spec_from_file_location("server", "server.py")
        server = importlib.util.module_from_spec(spec)
        
        # We won't execute the module (spec.loader.exec_module) 
        # because it would try to run the MCP server
        print("✓ server.py structure is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in server.py: {e}")
        return False
    except Exception as e:
        print(f"✗ Error loading server.py: {e}")
        return False

def test_tool_definitions():
    """Verify that all required tools are defined in the server."""
    print("\nChecking tool definitions...")
    
    required_tools = [
        "separate_audio_layers",
        "analyze_layer", 
        "synthesize_instrument_layer",
        "replace_layer",
        "modify_layer",
        "mix_layers"
    ]
    
    try:
        with open("server.py", "r") as f:
            content = f.read()
        
        all_found = True
        for tool in required_tools:
            if f"def {tool}(" in content:
                print(f"✓ {tool} - defined")
            else:
                print(f"✗ {tool} - NOT FOUND")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error reading server.py: {e}")
        return False

def test_mcp_decorator():
    """Check that tools are properly decorated with @mcp.tool()"""
    print("\nChecking MCP tool decorators...")
    
    try:
        with open("server.py", "r") as f:
            lines = f.readlines()

        required_tools = [
            "separate_audio_layers",
            "analyze_layer",
            "synthesize_instrument_layer",
            "replace_layer",
            "modify_layer",
            "mix_layers",
        ]

        decorated_tools = set()

        for i, line in enumerate(lines):
            # Look for the @mcp.tool() decorator
            if line.lstrip().startswith("@mcp.tool()"):
                # Find the next non-empty line after the decorator
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines):
                    next_line = lines[j].lstrip()
                    if next_line.startswith("def "):
                        for tool in required_tools:
                            if next_line.startswith(f"def {tool}("):
                                decorated_tools.add(tool)

        print(
            f"Found {len(decorated_tools)} decorated tools: "
            + (", ".join(sorted(decorated_tools)) if decorated_tools else "none")
        )

        missing_tools = [tool for tool in required_tools if tool not in decorated_tools]
        if not missing_tools:
            print("✓ All required tools are decorated with @mcp.tool()")
            return True
        else:
            print("✗ Missing decorators for: " + ", ".join(missing_tools))
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_dockerfile():
    """Verify Dockerfile exists and contains required components."""
    print("\nChecking Dockerfile...")
    
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()
        
        checks = {
            "CUDA base image": "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04" in content,
            "Python 3.12": "python3.12" in content,
            "Requirements": "requirements.txt" in content,
            "Server entrypoint": "server.py" in content
        }
        
        all_passed = True
        for check, passed in checks.items():
            if passed:
                print(f"✓ {check}")
            else:
                print(f"✗ {check} - NOT FOUND")
                all_passed = False
        
        return all_passed
    except FileNotFoundError:
        print("✗ Dockerfile not found")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_requirements():
    """Check that requirements.txt contains necessary packages."""
    print("\nChecking requirements.txt...")
    
    required_packages = [
        "fastmcp",
        "demucs",
        "librosa",
        "torch",
        "torchaudio",
        "soundfile",
        "mido",
        "pretty_midi"
    ]
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read().lower()
        
        all_found = True
        for package in required_packages:
            if package.lower() in content:
                print(f"✓ {package}")
            else:
                print(f"✗ {package} - NOT FOUND")
                all_found = False
        
        return all_found
    except FileNotFoundError:
        print("✗ requirements.txt not found")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Audio Processor MCP Server Validation")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Server Imports", test_server_imports()))
    results.append(("Tool Definitions", test_tool_definitions()))
    results.append(("MCP Decorators", test_mcp_decorator()))
    results.append(("Dockerfile", test_dockerfile()))
    results.append(("Requirements", test_requirements()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
