"""Request/response schemas for the GridFM PowerFlow-reconstruction plugin.

These are plain pydantic models with no vLLM dependency so they can be imported
and validated anywhere (client code, tests, the IO processor). A request carries
a single power-grid case as three record lists mirroring the columns the graph
builder consumes; the response carries per-node embeddings and the reconstructed
PowerFlow quantities.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class GridCase(BaseModel):
    """One power-grid scenario as column-record tables.

    Each list holds one record (dict) per row. Column names must match those
    consumed by :func:`gridfm_graphkit.datasets.graph_builder.build_hetero_data`
    (e.g. bus rows need ``bus``, ``Pd``, ``Qd`` ...; branch rows need
    ``from_bus``, ``to_bus``, ``pf`` ...). Buses must be listed in increasing
    ``bus`` order (0..N-1).
    """

    bus: list[dict[str, Any]] = Field(
        ...,
        description="Bus table records, ordered by increasing bus id (0..N-1).",
    )
    gen: list[dict[str, Any]] = Field(..., description="Generator table records.")
    branch: list[dict[str, Any]] = Field(..., description="Branch table records.")


class GridFMRequest(BaseModel):
    """Top-level request body for the /pooling endpoint of the GridFM plugin."""

    case: GridCase = Field(..., description="The power-grid case to run.")
    return_embeddings: bool = Field(
        True,
        description="Include per-node latent embeddings in the response.",
    )
    return_predictions: bool = Field(
        True,
        description="Include reconstructed Vm/Va/Pg predictions in the response.",
    )


class GridFMResponse(BaseModel):
    """Response body: reconstructed quantities and/or embeddings."""

    num_buses: int
    num_gens: int
    bus_predictions: Optional[list[list[float]]] = Field(
        None,
        description="Per-bus reconstructed outputs (Vm, Va, ...), denormalized.",
    )
    gen_predictions: Optional[list[list[float]]] = Field(
        None,
        description="Per-generator reconstructed outputs (Pg, ...), denormalized.",
    )
    bus_embeddings: Optional[list[list[float]]] = Field(
        None,
        description="Per-bus latent embeddings (head input of mlp_bus).",
    )
    gen_embeddings: Optional[list[list[float]]] = Field(
        None,
        description="Per-generator latent embeddings (head input of mlp_gen).",
    )
    request_id: Optional[str] = None
