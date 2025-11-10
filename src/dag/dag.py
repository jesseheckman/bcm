# src/dag/dag.py
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, Set, Iterable, Hashable, Optional, Any, List, Tuple, Iterator


class DirectedAcyclicGraph:
    def __init__(self, from_edges=None):
        self._nodes: Set[Hashable] = set()
        self._nodes_dict: Dict[Hashable, int] = dict()
        self._edges: Set[Hashable] = set()
        self._parents: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self._children: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self._graph: SignedTriangularAdjacencyMatrix = SignedTriangularAdjacencyMatrix(0)

        if from_edges is not None:
            self.from_edges(from_edges)
    

    def from_edges(self, edges: Iterable[Tuple[Hashable, Hashable]]) -> None:

        self._nodes = self._get_all_nodes(edges)                                    # Get the nodes from the edges
        self._nodes_dict = {node: idx for idx, node in enumerate(self._nodes)}      # Map nodes to indices
        self._graph = SignedTriangularAdjacencyMatrix(len(self._nodes))             # Initialize the adjacency matrix

        # Build edges
        for u, v in edges:
            self.add_edge(u, v)


    def _get_all_nodes(self, edges: Iterable[Tuple[Hashable, Hashable]]) -> Set[Hashable]:
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)

        print(f"\nNodes identified in graph: {nodes}")
        return nodes

    
    def add_node(self, node: Hashable) -> None:
        self._nodes.add(node)


    def add_edge(self, from_node: Hashable, to_node: Hashable) -> None:
        if from_node not in self._nodes or to_node not in self._nodes:
            raise ValueError("Both nodes must be added to the graph before adding an edge.")
        if self._creates_cycle(from_node, to_node):
            raise ValueError("Adding this edge would create a cycle.")
               
        self._edges.add((self._nodes_dict[from_node], self._nodes_dict[to_node]))
        self._parents[to_node].add(from_node)
        self._children[from_node].add(to_node)
        self._graph.set_edge(self._nodes_dict[from_node], self._nodes_dict[to_node], weight=1)
    

    def _creates_cycle(self, from_node: Hashable, to_node: Hashable) -> bool:
        visited = set()
        queue = deque([to_node])
        while queue:
            current = queue.popleft()
            if current == from_node:
                return True
            for parent in self._parents[current]:
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return False


    @staticmethod
    def _hi_lo(u: int, v: int) -> Tuple[int, int]:
        if u == v:
            raise ValueError("Self-loops are not allowed in a DAG.")
        return (u, v) if u > v else (v, u)


    def _get_cell(self, u: int, v: int) -> Tuple[int, int, int]:
        """Return (i, j, val) where i>j, val is stored signed integer."""
        i, j = self._hi_lo(u, v)
        return i, j, self._M[i][j]


    def clear_edge(self, u: int, v: int) -> None:
        i, j = self._hi_lo(u, v)
        self._graph[i][j] = 0


    def set_edge(self, u: int, v: int, weight: int = 1) -> None:
        """
        Create/overwrite a directed edge u -> v with (positive) weight.
        """
        if weight <= 0:
            raise ValueError("Weight must be a positive integer.")
        i, j = self._hi_lo(u, v)
        # + means j -> i; - means i -> j
        direction = "i_to_j" if u > v else "j_to_i"
        self._graph.set_edge(i, j, weight=weight, direction=direction)



    def has_edge(self, u: int, v: int) -> bool:
        i, j, val = self._get_cell(u, v)
        if val == 0:
            return False
        # decode direction
        if (u < v and val > 0) or (u > v and val < 0):
            return True  # u -> v encoded
        return False


    def weight(self, u: int, v: int) -> Optional[int]:
        """Return weight for edge u->v, or None if absent."""
        i, j, val = self._get_cell(u, v)
        if val == 0:
            return None
        # u -> v iff:
        if (u < v and val > 0) or (u > v and val < 0):
            return abs(val)
        return None


    def children(self, u: int) -> Iterator[int]:
        """Successors of u."""
        # case A: edges u -> v where v > u (stored at (v,u) with +)
        for v in range(u + 1, self.n):
            val = self._M[v][u]
            if val > 0:  # j->i encoded and j==u
                yield v
        # case B: edges u -> v where v < u (stored at (u,v) with -)
        row = self._M[u]
        for v in range(u):
            if row[v] < 0:
                yield v


    def parents(self, u: int) -> Iterator[int]:
        """Predecessors of u."""
        # case A: edges v -> u where v < u (stored at (u,v) with +)
        row = self._M[u]
        for v in range(u):
            if row[v] > 0:
                yield v
        # case B: edges v -> u where v > u (stored at (v,u) with -)
        for v in range(u + 1, self.n):
            val = self._M[v][u]
            if val < 0:
                yield v


    def edges(self) -> Iterator[Tuple[int, int, int]]:
        """Iterate (u, v, weight) for all directed edges."""
        for i in range(1, self.n):
            row = self._M[i]
            for j in range(i):
                val = row[j]
                if val == 0:
                    continue
                if val > 0:  # j -> i
                    yield (j, i, val)
                else:        # i -> j
                    yield (i, j, -val)




    def to_adjacency_list(self) -> List[List[Tuple[int, int]]]:
        """List-of-lists adjacency with (child, weight)."""
        adj = [[] for _ in range(self.n)]
        for u, v, w in self.edges():
            adj[u].append((v, w))
        return adj
    

    def is_acyclic(self) -> bool:
        """Kahn’s algorithm (O(n + m))."""
        indeg = [0] * self.n
        for _, v, _ in self.edges():
            indeg[v] += 1
        stack = [i for i, d in enumerate(indeg) if d == 0]
        seen = 0
        while stack:
            u = stack.pop()
            seen += 1
            for v in self.children(u):
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        return seen == self.n


class SignedTriangularAdjacencyMatrix:
    """
    Signed triangular adjacency for a directed acyclic graph on nodes {0..n-1}.
    Positive weights (+k) encode edges i -> j (where i > j),
    Negative weights (-k) encode edges j -> i (where i > j).
    Entries exist only for i > j.
    """

    def __init__(self, n: int):
        self.n = int(n)
        # Lower-triangular dense storage: row i has length i (columns 0..i-1)
        self._M: List[List[int]] = [[0 for _ in range(i)] for i in range(self.n)]


    def set_edge(self, i: int, j: int, weight: int = 1, direction: str = "i_to_j") -> None:
        if i == j:
            raise ValueError("Self-loops are not allowed in a DAG.")
        if weight <= 0:
            raise ValueError("Weight must be a positive integer.")

        hi, lo = (i, j) if i > j else (j, i)

        if direction not in ("i_to_j", "j_to_i"):
            raise ValueError("direction must be 'i_to_j' or 'j_to_i'.")

        # With hi > lo:
        # +k means hi → lo; −k means lo → hi
        self._M[hi][lo] = weight if direction == "i_to_j" else -weight


    def get_edge(self, i: int, j: int) -> int:
        """Return the signed weight between nodes i and j, or 0 if none."""
        if i == j:
            return 0
        hi, lo = (i, j) if i > j else (j, i)
        return self._M[hi][lo] if i > j else -self._M[hi][lo]


    def display(self) -> None:
        """Print the matrix in a readable lower-triangular form."""
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if i > j:
                    row.append(f"{self._M[i][j]:>3}")
                elif i == j:
                    row.append("  ·")  # diagonal placeholder
                else:
                    row.append("   ")  # upper empty
            print("".join(row))
        print()


    def __repr__(self) -> str:
        """Return a string representation of the matrix."""
        lines = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if i > j:
                    row.append(f"{self._M[i][j]:>3}")
                elif i == j:
                    row.append("  ·")
                else:
                    row.append("   ")
            lines.append("".join(row))
        return "\n".join(lines)


    def to_dense(self) -> List[List[int]]:
        """Return a full n×n matrix (with mirrored signed values)."""
        dense = [[0] * self.n for _ in range(self.n)]
        for i in range(1, self.n):
            for j in range(i):
                val = self._M[i][j]
                dense[i][j] = val
                dense[j][i] = -val
        return dense

    
# Allow alias for easier usage
DAG = DirectedAcyclicGraph
STAM = SignedTriangularAdjacencyMatrix

__all__ = ["DirectedAcyclicGraph", "DAG", "STAM", "SignedTriangularAdjacencyMatrix"]

if __name__ == "__main__":
    dag = DAG()

