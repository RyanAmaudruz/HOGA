"""
MIG (Majority-Inverter Graph) parser for Verilog files.

Extracts graphs from Verilog assign statements with boolean expressions,
detecting MAJ gates structurally and building a graph of nodes and edges.
"""

import re
from collections import defaultdict

# -------------------------
# AST Nodes
# -------------------------


class Var:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class Not:
    def __init__(self, child):
        self.child = child

    def __repr__(self):
        return f"~{repr(self.child)}"


class And:
    def __init__(self, children):
        self.children = children  # list

    def __repr__(self):
        return "&".join(sorted(repr(c) for c in self.children))


class Or:
    def __init__(self, children):
        self.children = children  # list

    def __repr__(self):
        return "|".join(sorted(repr(c) for c in self.children))


# -------------------------
# Verilog Expression Parser
# -------------------------


class ExprParser:
    def __init__(self, text):
        self.tokens = re.findall(r'\w+|[~&|()]', text)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self):
        return self.parse_or()

    def parse_or(self):
        node = self.parse_and()
        children = [node]
        while self.peek() == '|':
            self.consume()
            children.append(self.parse_and())
        if len(children) == 1:
            return node
        return Or(children)

    def parse_and(self):
        node = self.parse_unary()
        children = [node]
        while self.peek() == '&':
            self.consume()
            children.append(self.parse_unary())
        if len(children) == 1:
            return node
        return And(children)

    def parse_unary(self):
        tok = self.peek()
        if tok == '~':
            self.consume()
            return Not(self.parse_unary())
        elif tok == '(':
            self.consume()
            node = self.parse_or()
            assert self.consume() == ')'
            return node
        else:
            return Var(self.consume())


# -------------------------
# Canonicalization
# -------------------------


def flatten(node):
    if isinstance(node, And):
        children = []
        for c in node.children:
            c = flatten(c)
            if isinstance(c, And):
                children.extend(c.children)
            else:
                children.append(c)
        node.children = sorted(children, key=repr)
        return node

    if isinstance(node, Or):
        children = []
        for c in node.children:
            c = flatten(c)
            if isinstance(c, Or):
                children.extend(c.children)
            else:
                children.append(c)
        node.children = sorted(children, key=repr)
        return node

    if isinstance(node, Not):
        node.child = flatten(node.child)
        return node

    return node


# -------------------------
# Strict MAJ Detection
# -------------------------


def is_strict_majority(node):
    """
    Detect MAJ(a,b,c) structurally:
      OR of 3 AND terms
      Each AND has 2 distinct variables
      Exactly 3 unique variables total
      Each variable appears in exactly 2 AND terms
    """

    if not isinstance(node, Or):
        return False, None

    if len(node.children) != 3:
        return False, None

    and_terms = node.children
    if not all(isinstance(t, And) and len(t.children) == 2 for t in and_terms):
        return False, None

    pairs = []
    for t in and_terms:
        vars_in_term = []
        for c in t.children:
            if not isinstance(c, Var):
                return False, None
            vars_in_term.append(c.name)
        if vars_in_term[0] == vars_in_term[1]:
            return False, None
        pairs.append(tuple(sorted(vars_in_term)))

    all_vars = set(sum(pairs, ()))
    if len(all_vars) != 3:
        return False, None

    counts = defaultdict(int)
    for a, b in pairs:
        counts[a] += 1
        counts[b] += 1

    if not all(counts[v] == 2 for v in all_vars):
        return False, None

    return True, sorted(all_vars)


# -------------------------
# Graph Builder (MIG)
# -------------------------


class MIGGraphBuilder:
    def __init__(self):
        self.gate_id = 0
        self.not_id = 0
        self.nodes = {}
        self.edges = []

    def new_gate(self, gtype):
        gid = f"g{self.gate_id}"
        self.gate_id += 1
        self.nodes[gid] = gtype
        return gid

    def new_not(self):
        nid = f"not{self.not_id}"
        self.not_id += 1
        self.nodes[nid] = "NOT"
        return nid

    def ensure_net(self, name):
        if name not in self.nodes:
            self.nodes[name] = "net"

    def build_from_assign(self, lhs, expr):
        parser = ExprParser(expr)
        ast = flatten(parser.parse())

        self.ensure_net(lhs)

        self._build_recursive(lhs, ast)

    def _build_recursive(self, out_net, node):

        # Strict MAJ detection
        is_maj, vars3 = is_strict_majority(node)
        if is_maj:
            gid = self.new_gate("MAJ")
            for v in vars3:
                self.ensure_net(v)
                self.edges.append((v, gid))
            self.edges.append((gid, out_net))
            return

        if isinstance(node, Var):
            # BUF
            gid = self.new_gate("BUF")
            self.ensure_net(node.name)
            self.edges.append((node.name, gid))
            self.edges.append((gid, out_net))
            return

        if isinstance(node, Not):
            child = node.child
            if isinstance(child, Var):
                not_id = self.new_not()
                self.ensure_net(child.name)
                self.edges.append((child.name, not_id))
                self.edges.append((not_id, out_net))
            else:
                tmp = f"_tmp_{self.gate_id}"
                self.ensure_net(tmp)
                self._build_recursive(tmp, child)
                not_id = self.new_not()
                self.edges.append((tmp, not_id))
                self.edges.append((not_id, out_net))
            return

        if isinstance(node, And):
            gid = self.new_gate("AND")
            for c in node.children:
                tmp = self._ensure_child(c)
                self.edges.append((tmp, gid))
            self.edges.append((gid, out_net))
            return

        if isinstance(node, Or):
            gid = self.new_gate("OR")
            for c in node.children:
                tmp = self._ensure_child(c)
                self.edges.append((tmp, gid))
            self.edges.append((gid, out_net))
            return

    def _ensure_child(self, node):
        if isinstance(node, Var):
            self.ensure_net(node.name)
            return node.name

        tmp = f"_tmp_{self.gate_id}"
        self.ensure_net(tmp)
        self._build_recursive(tmp, node)
        return tmp


# -------------------------
# Top-Level Verilog Parser
# -------------------------


def parse_verilog_file(path):
    builder = MIGGraphBuilder()

    with open(path) as f:
        lines = f.readlines()

    assign_re = re.compile(r"assign\s+(\w+)\s*=\s*(.+);")

    for line in lines:
        m = assign_re.match(line.strip())
        if m:
            lhs, rhs = m.groups()
            rhs = rhs.strip()
            builder.build_from_assign(lhs, rhs)

    return builder.nodes, builder.edges
