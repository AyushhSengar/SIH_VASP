"""
Output models for the graph layer.

Kept separate from app/models.py (the provider-facing NormalizedTransfer
schema) since graph summaries are a derived/analysis-layer concept, not
part of the canonical transaction schema.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DegreeEntry(BaseModel):
    address: str
    degree: int


class GraphBuildStats(BaseModel):
    """Authoritative bookkeeping produced directly by build_graph() itself —
    never re-derived from free-text notes. This is the single source of
    truth for the input/output accounting equation.
    """

    input_transfer_count: int
    edges_created: int
    contract_creation_skipped: int
    other_skipped: int = 0
    notes: list[str] = []

    @property
    def accounted_for(self) -> int:
        return self.edges_created + self.contract_creation_skipped + self.other_skipped

    @property
    def reconciled(self) -> bool:
        return self.accounted_for == self.input_transfer_count


class GraphSummary(BaseModel):
    input_transfer_count: int
    edges_created: int  # from GraphBuildStats bookkeeping
    edge_count: int  # independently counted via graph.number_of_edges() — cross-check

    node_count: int

    native_edge_count: int
    token_edge_count: int

    contract_creation_skipped: int
    other_skipped: int
    self_loop_edges: int

    accounted_for: int
    reconciled: bool

    average_out_degree: float
    average_in_degree: float

    top_out_degree_nodes: list[DegreeEntry]
    top_in_degree_nodes: list[DegreeEntry]

    density: float

    earliest_timestamp: Optional[int] = None
    latest_timestamp: Optional[int] = None

    notes: list[str] = []
