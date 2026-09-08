"""
ANALYSIS ROUTE — the real-data investigation surface.

`POST /analysis/brief` runs the same pipeline `investigate.py` runs and returns
the same `InvestigationBrief` that `--brief` prints. One code path, one
projection: the browser cannot be shown a field the terminal does not have, and
neither can drift into a different verdict for the same wallet.

WHY THIS IS A NEW ROUTER AND NOT AN EXTRA FIELD ON /investigations
--------------------------------------------------------------------------
`/investigations` is the Milestone-6 persistence surface. Its ML summary is
type-locked to the synthetic demonstration classifier
(`training_data_type="SYNTHETIC_DEMO"`), which callers and tests both rely on.
Swapping the model underneath it would silently change the meaning of a field
that is documented and asserted elsewhere. This router serves the real-data ML
pipeline instead, and the two remain distinguishable by the endpoint you call.

NO PERSISTENCE HERE
--------------------------------------------------------------------------
This route computes and returns; it does not write an InvestigationRecord.
Persisting the brief would mean storing a *projection* as though it were the
finding, and the finding is the full report. `/investigations` remains the
endpoint that records an investigation.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.api.schemas import BriefRequest
from app.core.config import get_settings
from app.investigation.runner import run_wallet_investigation
from app.reporting.brief import InvestigationBrief, build_brief

logger = logging.getLogger("app.api")

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/brief", response_model=InvestigationBrief)
async def post_brief(body: BriefRequest) -> InvestigationBrief:
    """Investigates one wallet and returns the at-a-glance brief.

    Runs synchronously inside the request: there is no job queue, because a
    queue would need a second endpoint to poll, a store for partial results and
    a story for what a half-finished investigation means. A live fetch expands
    hop by hop and takes minutes, which is why `prefer_cached` defaults to
    reusing a real artefact when one exists for the wallet — labelled CACHED
    REAL DATA in the response, never as a live result.
    """
    settings = get_settings()
    report, choice = await run_wallet_investigation(
        body.wallet,
        body.chain,
        settings,
        prefer_cached=body.prefer_cached,
        max_hops=body.max_hops,
        max_paths=body.max_paths,
        enable_ml=body.enable_ml,
    )
    # Server-side only: which artefact answered which request is an operational
    # fact, and the response already carries the data mode the reader needs.
    logger.info(
        "brief for %s on %s via %s (%s)",
        report.wallet,
        report.chain,
        choice.mode.value,
        report.provenance.data_mode.value,
    )
    return build_brief(report)
