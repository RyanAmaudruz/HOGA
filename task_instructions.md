Update the mig_parser.py file to this new improved parser:

"""
import re

# -------------------------
# 1. AST Nodes (Simplified)
# -------------------------
class Var:
    def __init__(self, name): self.name = name
    def __repr__(self): return self.name

class Const:
    def __init__(self, val): self.val = val # 0 or 1
    def __repr__(self): return str(self.val)

class Not:
    def __init__(self, child): self.child = child
    def __repr__(self): return f"~{self.child}"

class And:
    def __init__(self, children): self.children = children
    def __repr__(self): return f"({' & '.join(map(str, self.children))})"

class Or:
    def __init__(self, children): self.children = children
    def __repr__(self): return f"({' | '.join(map(str, self.children))})"

# -------------------------
# 2. Expression Parser
# -------------------------
class ExprParser:
    def __init__(self, text):
        # Improved regex to catch 1'b0, 1'b1, 0, 1
        self.tokens = re.findall(r"1'b[01]|1'd[01]|\b\d+\b|\w+|[~&|()]", text)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self): return self.parse_or()

    def parse_or(self):
        node = self.parse_and()
        children = [node]
        while self.peek() == '|':
            self.consume()
            children.append(self.parse_and())
        return Or(children) if len(children) > 1 else node

    def parse_and(self):
        node = self.parse_unary()
        children = [node]
        while self.peek() == '&':
            self.consume()
            children.append(self.parse_unary())
        return And(children) if len(children) > 1 else node

    def parse_unary(self):
        tok = self.peek()
        if tok == '~':
            self.consume()
            return Not(self.parse_unary())
        elif tok == '(':
            self.consume()
            node = self.parse_or()
            self.consume() # ')'
            return node
        elif tok in ['0', "1'b0", "1'd0"]:
            self.consume()
            return Const(0)
        elif tok in ['1', "1'b1", "1'd1"]:
            self.consume()
            return Const(1)
        else:
            return Var(self.consume())

# -------------------------
# 3. Canonical MIG Builder
# -------------------------
class MIGBuilder:
    def __init__(self):
        # Node storage: 0=Const0, 1=Const1
        # Everything else is a MAJ gate or PI (Primary Input)
        self.node_count = 2 
        
        # Maps wire_name -> (node_index, is_inverted)
        self.wire_map = {
            '0': (0, False), "1'b0": (0, False),
            '1': (1, False), "1'b1": (1, False)
        }
        
        # Graph Data for PyG
        # nodes: list of node types (0=Const, 1=PI, 2=MAJ)
        self.node_types = {0: 'CONST', 1: 'CONST'} 
        
        # edges: list of (source, target, is_inverted)
        self.edges = null 

    def get_wire(self, name):
        """Register a PI (Primary Input) if seen for first time."""
        if name not in self.wire_map:
            idx = self.node_count
            self.node_count += 1
            self.wire_map[name] = (idx, False)
            self.node_types[idx] = 'PI'
        return self.wire_map[name]

    def add_maj_gate(self, op1, op2, op3):
        """
        Creates a generic MAJ gate.
        Inputs: tuples of (node_idx, is_inverted)
        Returns: (new_node_idx, is_inverted=False)
        """
        new_idx = self.node_count
        self.node_count += 1
        self.node_types[new_idx] = 'MAJ'

        # Add 3 edges
        for (src_idx, inv) in [op1, op2, op3]:
            self.edges.append({
                'src': src_idx,
                'dst': new_idx,
                'inverted': inv
            })
            
        return (new_idx, False)

    def process_expression(self, node):
        """Recursively converts AST -> MAJ Nodes."""
        
        if isinstance(node, Var):
            return self.get_wire(node.name)
            
        if isinstance(node, Const):
            return (0, False) if node.val == 0 else (1, False)

        if isinstance(node, Not):
            idx, inv = self.process_expression(node.child)
            return (idx, not inv) # Flip polarity (Edge Attribute)

        if isinstance(node, And):
            # Decompose n-ary AND: AND(a,b,c) -> AND(a, AND(b,c))
            # Base case: AND(a,b) -> MAJ(a, b, 0)
            children = [self.process_expression(c) for c in node.children]
            curr = children[0]
            for next_op in children[1:]:
                # MAJ(a, b, 0)
                curr = self.add_maj_gate(curr, next_op, (0, False))
            return curr

        if isinstance(node, Or):
            # Decompose n-ary OR
            # Base case: OR(a,b) -> MAJ(a, b, 1)
            children = [self.process_expression(c) for c in node.children]
            curr = children[0]
            for next_op in children[1:]:
                # MAJ(a, b, 1)
                curr = self.add_maj_gate(curr, next_op, (1, False))
            return curr
            
        raise ValueError(f"Unknown node: {node}")

    def add_assign(self, lhs, rhs_expr_str):
        parser = ExprParser(rhs_expr_str)
        ast = parser.parse()
        
        # Recursively build graph
        result_idx, result_inv = self.process_expression(ast)
        
        # Register the LHS name as an alias to the result node
        # This handles buffers implicitly!
        self.wire_map[lhs] = (result_idx, result_inv)

# -------------------------
# 4. Interface
# -------------------------
def parse_verilog_to_mig(path):
    builder = MIGBuilder()
    
    with open(path) as f:
        lines = f.readlines()
        
    assign_re = re.compile(r"assign\s+(\w+)\s*=\s*(.+);")
    
    # First pass: Identify all PIs (inputs) that might appear on RHS
    # In real Verilog parsers we read 'input' decls, but here we infer.
    
    for line in lines:
        m = assign_re.match(line.strip())
        if m:
            lhs, rhs = m.groups()
            builder.add_assign(lhs, rhs.strip())

    return builder

"""


And also add a new python file containing this (adjust where needed):
"""
# Convert Builder output to PyG
import torch
from torch_geometric.data import Data

def builder_to_pyg(builder):
    # 1. Node Features
    # Map types to ints: CONST=0, PI=1, MAJ=2
    type_map = {'CONST': 0, 'PI': 1, 'MAJ': 2}
    x = torch.tensor([type_map[builder.node_types[i]] for i in range(builder.node_count)], dtype=torch.long)
    # If you want One-Hot: F.one_hot(x, num_classes=3)

    # 2. Edges
    src = null
    dst = null
    attrs = null # 0=Normal, 1=Inverted
    
    for e in builder.edges:
        src.append(e['src'])
        dst.append(e['dst'])
        attrs.append(1 if e['inverted'] else 0)
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
"""

Finally, make sure the demo_mig_parser.py file still runs.