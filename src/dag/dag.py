# src/dag/dag.py
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, Set, Iterable, Hashable, Optional, Any, List, Tuple


# ---------- Exceptions ----------
class CycleError(Exception): ...
class NodeNotFoundError(KeyError): ...
class EdgeExistsError(ValueError): ...
class EdgeNotFoundError(KeyError): ...


# ---------- Optional structured metadata ----------
@dataclass(slots=True)
class NodeAttr:
    kind: Optional[str] = None      # e.g. "observed" | "latent" | "intervention"
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class EdgeAttr:
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------- Core DAG (mutable) ----------
class DAG:
    def __init__(self):
        self._nodes: Set[Hashable] = set()
        self._parents: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self._children: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self._node_attr: Dict[Hashable, NodeAttr] = {}
        self._edge_attr: Dict[Tuple[Hashable, Hashable], EdgeAttr] = {}

    # --- Node ops ---
    def add_node(self, node: Hashable, attr: Optional[NodeAttr] = None) -> None:
        self._nodes.add(node)
        if attr is not None:
            self._node_attr[node] = attr

    def remove_node(self, node: Hashable) -> None:
        if node not in self._nodes:
            raise NodeNotFoundError(node)
        # remove incident edges
        for p in list(self._parents[node]): self.remove_edge(p, node)
        for c in list(self._children[node]): self.remove_edge(node, c)
        self._nodes.remove(node)
        self._node_attr.pop(node, None)

    # --- Edge ops ---
    def add_edge(self, parent: Hashable, child: Hashable, attr: Optional[EdgeAttr] = None) -> None:
        # auto-add nodes
        if parent not in self._nodes: self.add_node(parent)
        if child not in self._nodes: self.add_node(child)

        if child in self._children[parent]:
            raise EdgeExistsError((parent, child))

        # tentatively add, then cycle-check
        self._parents[child].add(parent)
        self._children[parent].add(child)
        if not self.is_acyclic():
            # rollback
            self._parents[child].remove(parent)
            self._children[parent].remove(child)
            raise CycleError(f"Adding {parent}->{child} creates a cycle")

        if attr is not None:
            self._edge_attr[(parent, child)] = attr

    def remove_edge(self, parent: Hashable, child: Hashable) -> None:
        if child not in self._children.get(parent, set()):
            raise EdgeNotFoundError((parent, child))
        self._children[parent].remove(child)
        self._parents[child].remove(parent)
        self._edge_attr.pop((parent, child), None)

    # --- Queries ---
    @property
    def nodes(self) -> Set[Hashable]:
        return set(self._nodes)

    def parents(self, node: Hashable) -> Set[Hashable]:
        if node not in self._nodes: raise NodeNotFoundError(node)
        return set(self._parents[node])

    def children(self, node: Hashable) -> Set[Hashable]:
        if node not in self._nodes: raise NodeNotFoundError(node)
        return set(self._children[node])

    def has_edge(self, u: Hashable, v: Hashable) -> bool:
        return v in self._children.get(u, set())

    # --- Attributes ---
    def get_node_attr(self, node: Hashable) -> NodeAttr:
        if node not in self._nodes: raise NodeNotFoundError(node)
        return self._node_attr.get(node, NodeAttr())

    def set_node_attr(self, node: Hashable, **kwargs) -> None:
        cur = self.get_node_attr(node)
        for k, v in kwargs.items():
            setattr(cur, k, v)
        self._node_attr[node] = cur

    def get_edge_attr(self, parent: Hashable, child: Hashable) -> EdgeAttr:
        if not self.has_edge(parent, child): raise EdgeNotFoundError((parent, child))
        return self._edge_attr.get((parent, child), EdgeAttr())

    def set_edge_attr(self, parent: Hashable, child: Hashable, **kwargs) -> None:
        cur = self.get_edge_attr(parent, child)
        for k, v in kwargs.items():
            setattr(cur, k, v)
        self._edge_attr[(parent, child)] = cur

    # --- Algorithms ---
    def is_acyclic(self) -> bool:
        indeg = {n: len(self._parents[n]) for n in self._nodes}
        q = deque([n for n in self._nodes if indeg[n] == 0])
        seen = 0
        while q:
            u = q.popleft(); seen += 1
            for v in self._children[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        return seen == len(self._nodes)

    def topo_order(self) -> List[Hashable]:
        indeg = {n: len(self._parents[n]) for n in self._nodes}
        q = deque([n for n in self._nodes if indeg[n] == 0])
        order: List[Hashable] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self._children[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        if len(order) != len(self._nodes):
            raise CycleError("Graph is not acyclic.")
        return order

    # --- Serialisation (structure + light metadata only) ---
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"id": n, "attr": self._node_attr.get(n, NodeAttr()).__dict__}
                for n in sorted(self._nodes, key=str)
            ],
            "edges": [
                {"u": u, "v": v, "attr": self._edge_attr.get((u, v), EdgeAttr()).__dict__}
                for u in self._nodes for v in self._children[u]
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DAG":
        dag = cls()
        for n in payload.get("nodes", []):
            dag.add_node(n["id"], NodeAttr(**n.get("attr", {})))
        for e in payload.get("edges", []):
            dag.add_edge(e["u"], e["v"], EdgeAttr(**e.get("attr", {})))
        return dag

    # --- Views ---
    def freeze(self) -> "FrozenDAG":
        return FrozenDAG(self)

    def subdag(self, nodes: Iterable[Hashable]) -> "SubDAG":
        return SubDAG(self, set(nodes))


# ---------- Read-only snapshot ----------
class FrozenDAG:
    def __init__(self, dag: DAG):
        self._nodes = frozenset(dag.nodes)
        self._edges = frozenset((u, v) for u in dag.nodes for v in dag.children(u))
        self._topo = tuple(dag.topo_order())

    @property
    def nodes(self) -> Set[Hashable]:
        return set(self._nodes)

    @property
    def edges(self) -> Set[Tuple[Hashable, Hashable]]:
        return set(self._edges)

    def topo_order(self) -> List[Hashable]:
        return list(self._topo)


# ---------- Read-only view on a subset ----------
class SubDAG:
    def __init__(self, base: DAG, keep: Set[Hashable]):
        missing = keep.difference(base.nodes)
        if missing:
            raise NodeNotFoundError(f"Unknown nodes in subdag: {missing}")
        self._base = base
        self._keep = set(keep)

    @property
    def nodes(self) -> Set[Hashable]:
        return set(self._keep)

    def children(self, node: Hashable) -> Set[Hashable]:
        return {c for c in self._base.children(node) if c in self._keep}

    def parents(self, node: Hashable) -> Set[Hashable]:
        return {p for p in self._base.parents(node) if p in self._keep}

    def topo_order(self) -> List[Hashable]:
        # simple Kahn on the induced subgraph
        indeg = {n: len(self.parents(n)) for n in self._keep}
        from collections import deque
        q = deque([n for n, d in indeg.items() if d == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.children(u):
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        if len(order) != len(self._keep):
            raise CycleError("Subgraph is not acyclic.")
        return order
