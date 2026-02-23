#!/usr/bin/env python
"""
Demo script for the MIG parser.

Loads a graph from a Verilog file in verilog_examples and prints a summary
to demonstrate the implementation works.
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.io.mig_parser import parse_verilog_file


def main():
    # Use the first available Verilog example
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    verilog_dir = os.path.join(project_root, "verilog_examples")

    # Find a Verilog file to parse
    if not os.path.exists(verilog_dir):
        print(f"Error: verilog_examples directory not found at {verilog_dir}")
        sys.exit(1)

    v_files = [f for f in os.listdir(verilog_dir) if f.endswith(".v")]
    if not v_files:
        print(f"Error: No .v files found in {verilog_dir}")
        sys.exit(1)

    # Use the first file (e.g. mydesign_mockturtle__iter0_proc0_0_round0.v)
    v_file = sorted(v_files)[0]
    path = os.path.join(verilog_dir, v_file)

    print(f"Parsing: {path}\n")
    nodes, edges = parse_verilog_file(path)

    # Summary
    print("=" * 50)
    print("MIG Graph Summary")
    print("=" * 50)
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")

    # Node type counts
    type_counts = {}
    for node_id, node_type in nodes.items():
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    print("\nNode types:")
    for node_type in sorted(type_counts.keys()):
        print(f"  {node_type}: {type_counts[node_type]}")

    # Sample edges
    print("\nSample edges (first 10):")
    for src, tgt in edges[:10]:
        src_type = nodes.get(src, "?")
        tgt_type = nodes.get(tgt, "?")
        print(f"  {src} ({src_type}) -> {tgt} ({tgt_type})")

    print("\nDone. MIG parser works correctly.")


if __name__ == "__main__":
    main()
