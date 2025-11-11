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
        self._coparents: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self._children: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self._graph: SignedTriangularAdjacencyMatrix = SignedTriangularAdjacencyMatrix(0)
        self.config: DagConfig = DagConfig()

        if from_edges is not None:
            self.from_edges(from_edges)
    

    def from_edges(self, edges: Iterable[Tuple[Hashable, Hashable]]) -> None:
        self._nodes = self._get_nodes_from_edges(edges)                                                         # Get the nodes from the edges
        self._nodes_dict = {node: idx for idx, node in enumerate(self.get_nodes(srt_method="str_sort"))}        # Map nodes to indices alphabetically
        self._graph = SignedTriangularAdjacencyMatrix(len(self._nodes))                                         # Initialize the STAM

        # Build edges
        for u, v in edges:
            self.add_edge(u, v)


    def _get_nodes_from_edges(self, edges: Iterable[Tuple[Hashable, Hashable]]) -> Set[Hashable]:
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)

        return nodes
    

    def get_nodes(self, srt_method: Optional[str] = None) -> List[Hashable]:
        """Return all nodes sorted according to the specified method.

        Parameters:
        srt_method : str, optional
        """
        
        # Default sorting method
        if srt_method is None:
            srt_method = getattr(getattr(self, "config", {}), "sort_method", "str_sort")

        print(f" > Getting nodes with sort method: {srt_method}")
        
        # Sorting methods
        if srt_method in ("str_sort", "str", "alphabetical"):
            return sorted(self._nodes, key=str)
        elif srt_method in ("topo", "topo_sort", "topologic_sort", "topological_sort"):
            return self._topologic_sort()
        else:
            raise ValueError(f"Unknown sort method: {srt_method}")

    
    def add_node(self, node: Hashable) -> None:
        self._nodes.add(node)


    def add_edge(self, from_node: Hashable, to_node: Hashable) -> None:
        if from_node not in self._nodes or to_node not in self._nodes:
            raise ValueError("Both nodes must be added to the graph before adding an edge.")
        if self._creates_cycle(from_node, to_node):
            raise ValueError("Adding this edge would create a cycle.")
               
        self._edges.add((self._nodes_dict[from_node], self._nodes_dict[to_node]))
        self._graph.set_edge(self._nodes_dict[from_node], self._nodes_dict[to_node], weight=1)
        self._fill_genology(from_node, to_node)


    def _fill_genology(self, from_node: Hashable, to_node: Hashable) -> None:
        """Fill in the _parents, _children, and _co_parents dictionaries based on current edges."""

        u = self._nodes_dict[from_node]  # index of from_node
        v = self._nodes_dict[to_node]    # index of to_node

        # Update parent/child relations
        self._parents[v].add(u)
        self._children[u].add(v)

        # Update co-parent relations
        for existing_parent in self._parents[v]:
            if existing_parent == u:
                continue
            # They share a child, so mark each other as co-parents
            self._coparents[u].add(existing_parent)
            self._coparents[existing_parent].add(u)


    def _creates_cycle(self, from_node: Hashable, to_node: Hashable) -> bool:
        """Return True if adding (from_node -> to_node) would create a cycle."""
        try:
            u = self._nodes_dict[from_node]  # index of from_node
            v = self._nodes_dict[to_node]    # index of to_node
        except KeyError:
            raise ValueError("Both nodes must be added to the graph before cycle check.")

        if u == v:  # self-loop
            return True

        # If there is already a path v -> ... -> u via _children, adding u -> v makes a cycle.
        seen = set([v])
        q = deque([v])
        while q:
            cur = q.popleft()
            if cur == u:
                return True
            for child in self._children[cur]:
                if child not in seen:
                    seen.add(child)
                    q.append(child)
        return False
    

    def _topologic_sort(self) -> List[Hashable]:
        """
        Kahn's algorithm for topological sorting.
        Returns the nodes (original hashables) in topological order.
        Raises ValueError if a cycle is detected.
        """
        if not self._nodes:
            return []

        # Map indices <-> original node labels
        idx_to_node: Dict[int, Hashable] = {idx: node for node, idx in self._nodes_dict.items()}

        n = len(self._nodes)

        # Compute indegrees from the parent sets (all stored as indices)
        indeg = [0] * n
        for v_idx, parents in self._parents.items():
            indeg[v_idx] = len(parents)

        # Use a heap to obtain deterministic order (alphabetical by node label string)
        import heapq
        heap: List[Tuple[str, int]] = []
        for i in range(n):
            if indeg[i] == 0:
                heapq.heappush(heap, (str(idx_to_node[i]), i))

        order_indices: List[int] = []

        while heap:
            _, u = heapq.heappop(heap)
            order_indices.append(u)

            # Decrease indegree of children; push those that become zero
            for v in list(self._children[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(heap, (str(idx_to_node[v]), v))

        if len(order_indices) != n:
            # Not all nodes were processed -> cycle present
            raise ValueError("Graph contains a cycle; topological sort is undefined.")

        # Return original node labels in topo order
        return [idx_to_node[i] for i in order_indices]


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


    def set_edge(self, i: int, j: int, weight: int = 1) -> None:
        if i == j:
            raise ValueError("Self-loops are not allowed in a DAG.")
        if weight <= 0:
            raise ValueError("Weight must be a positive integer.")

        hi, lo = (i, j) if i > j else (j, i)

        if i < j:
            weight = - weight  # i → j

        # With hi > lo:
        # +k means hi → lo; −k means lo → hi
        self._M[hi][lo] = weight


    def get_edge(self, i: int, j: int) -> int:
        """Return the signed weight between nodes i and j, or 0 if none."""
        if i == j:
            return 0
        hi, lo = (i, j) if i > j else (j, i)
        return self._M[hi][lo] if i > j else -self._M[hi][lo]


    def display(self) -> None:
        """Print the matrix in a readable lower-triangular form."""
        print("\n")
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


class DagConfig:
    """Configuration settings for DirectedAcyclicGraph."""
    def __init__(self):
        self.sort_method: str = "topologic_sort"
        self.edge_weight: int = 1

    
# Allow alias for easier usage
DAG = DirectedAcyclicGraph
STAM = SignedTriangularAdjacencyMatrix

__all__ = ["DirectedAcyclicGraph", "DAG", "STAM", "SignedTriangularAdjacencyMatrix"]

if __name__ == "__main__":
    dag = DAG()

