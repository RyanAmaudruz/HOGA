"""
Convert MIGBuilder output to PyTorch Geometric Data format.
"""

import torch
from torch_geometric.data import Data


def builder_to_pyg(builder):
    """Convert MIGBuilder output to PyG Data format."""
    # 1. Node Features
    # Map types to ints: CONST=0, PI=1, MAJ=2
    type_map = {"CONST": 0, "PI": 1, "MAJ": 2}
    x = torch.tensor(
        [type_map[builder.node_types[i]] for i in range(builder.node_count)],
        dtype=torch.long,
    )

    # 2. Edges
    src = []
    dst = []
    attrs = []  # 0=Normal, 1=Inverted

    for e in builder.edges:
        src.append(e["src"])
        dst.append(e["dst"])
        attrs.append(1 if e["inverted"] else 0)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float).view(-1, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
