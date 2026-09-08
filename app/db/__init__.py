"""
MACRO MILESTONE 6 — persistence layer.

Isolated from the blockchain-intelligence logic (app/graph, app/tracing,
app/behavior, app/attribution, app/ml): nothing in those packages imports
from here, and nothing here imports domain logic — only the pydantic
*output* models (AttributionResult, MLPrediction) for serialization.
"""
