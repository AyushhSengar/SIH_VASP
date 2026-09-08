"""
MACRO MILESTONE 6 — InvestigationService.

Orchestrates the EXISTING M1-M5 pipeline end to end and persists the
result. Every blockchain-intelligence step below is a call into an
already-existing M1-M5 function — nothing here re-implements graph
building, tracing, behavior detection, attribution, or ML.

Pipeline (unchanged order, per the M6 spec):

    EtherscanProvider
    -> acquire_wallet_transactions   (native + internal + token streams)
    -> normalize_all
    -> build_graph
    -> trace_fund_flow
    -> analyze_wallet_behavior
    -> generate_candidates
    -> extract_wallet_features
    -> train_model
    -> predict

ACQUISITION (fixed)
--------------------------------------------------------------------------
Acquisition delegates to `app.blockchain.ingest.acquire_wallet_transactions`,
the same function the CLI uses. It previously used a local pagination loop
that (a) never requested internal transactions, so value moved by contract
calls was missing from every graph this service built, and (b) caught
RateLimitError and ProviderAPIError and simply stopped, producing a partial
graph that was then reported as if it were the wallet's whole history. A
silently truncated dataset is worse than an error, because a "no VASP
connection" answer computed from half the transactions looks identical to a
real negative. Incompleteness is now carried on the result and surfaced in
`incompleteness_reasons`.

ML NOTE (deliberate, and stated rather than hidden)
--------------------------------------------------------------------------
This service still calls the Milestone-5 demonstration classifier, which is
trained on the synthetic dataset in `app/ml/training_data.py`. That model is
type-locked to `training_data_type="SYNTHETIC_DEMO"` and ships a disclaimer,
so it cannot be mistaken for a real-data result by any caller. The
production ML engine is the real-data one in `app/ml/real_training.py`, used
by `investigate.py`; this HTTP surface has not been migrated to it, and that
gap is recorded in README.md rather than papered over here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.attribution.candidate_generator import generate_candidates
from app.attribution.matcher import build_seed_index
from app.attribution.models import AttributionResult
from app.attribution.seed_loader import SeedDataError, load_vasp_seed
from app.behavior.detectors import analyze_wallet_behavior
from app.blockchain.base import (
    BlockchainProvider,
    InvalidAddressError,
    ProviderUnavailableError,
)
from app.blockchain.base import UnsupportedChainError as ProviderChainMismatch

# Chains this deployment accepts, taken from the provider-layer registry so the
# API and the CLI cannot drift apart. app.blockchain.chains is the single place
# that maps a chain name to a chain id; a name absent from it has no resolvable
# id, and accepting it would mean querying one chain and labelling the results
# as another. `normalize_chain_name` is imported alongside so this surface
# forgives the same things the CLI does (case, surrounding whitespace) and
# nothing more.
from app.blockchain.chains import SUPPORTED_CHAINS, normalize_chain_name
from app.blockchain.etherscan import EtherscanProvider
from app.blockchain.ingest import IngestionError, acquire_wallet_transactions
from app.core.config import Settings
from app.db.models import InvestigationRecord
from app.db.repository import InvestigationRepository
from app.graph.builder import build_graph, save_graph
from app.investigation.errors import (
    GraphNotFoundError,
    InternalServiceFailure,
    InvalidWalletError,
    UnsupportedChainError,
)
from app.ml.features import extract_wallet_features
from app.ml.models import MLPrediction
from app.ml.predictor import DEFAULT_SEED, predict, train_model
from app.normalization.transactions import normalize_all
from app.tracing.tracer import trace_fund_flow

# NOTE (do not re-hardcode): the graph cache directory is resolved from
# Settings.graph_cache_dir (GRAPH_CACHE_DIR), never from a module-level
# constant. A hardcoded "data/graphs" here meant the test suite wrote
# throwaway graphs into the *production* cache directory, which is a
# reproducibility hazard: a later real investigation could load a graph a
# test had left behind. Tests point GRAPH_CACHE_DIR at a tmp_path instead.

ProviderFactory = Callable[[Settings, str], BlockchainProvider]


def _default_provider_factory(settings: Settings, chain: str) -> BlockchainProvider:
    return EtherscanProvider(settings, chain_name=chain)


@dataclass(frozen=True)
class InvestigationRunResult:
    """Everything the API layer needs to build both the POST /investigations
    response and the persisted InvestigationRecord — a plain bundle, not a
    persistence or transport model itself."""

    investigation_id: str
    wallet: str
    chain: str
    max_hops: int
    max_paths: int
    graph_path: Optional[str]
    attribution: AttributionResult
    ml_prediction: MLPrediction
    record: InvestigationRecord
    #: False when any transaction stream was cut short by a provider failure
    #: or by MAX_TRANSACTIONS_PER_INVESTIGATION. A caller must read a negative
    #: finding on an incomplete dataset as inconclusive, never as proven absent.
    data_complete: bool = True
    incompleteness_reasons: list[str] = field(default_factory=list)


class InvestigationService:
    def __init__(
        self,
        settings: Settings,
        repository: InvestigationRepository,
        provider_factory: ProviderFactory = _default_provider_factory,
    ):
        self._settings = settings
        self._repository = repository
        self._provider_factory = provider_factory

    async def run_investigation(
        self,
        *,
        wallet: str,
        chain: str = "ethereum",
        max_hops: Optional[int] = None,
        max_paths: Optional[int] = None,
        use_demo_seed: bool = False,
        ml_seed: Optional[int] = None,
    ) -> InvestigationRunResult:
        # 1. Validate wallet address and chain.
        # The name is normalized BEFORE the membership test and then used for
        # the rest of the run, so the chain written onto every transfer, stored
        # on the record and echoed in the response is the canonical name the
        # provider was actually queried with — not the caller's capitalisation.
        requested_chain = chain
        chain = normalize_chain_name(chain)
        if chain not in SUPPORTED_CHAINS:
            raise UnsupportedChainError(
                f"Unsupported chain '{requested_chain}'. Supported chains: "
                f"{sorted(SUPPORTED_CHAINS)}."
            )

        try:
            provider = self._provider_factory(self._settings, chain)
        except ProviderUnavailableError as exc:
            # A missing credential is a configuration fault, not a network
            # fault, and must not be reported as "the provider could not be
            # reached" — that sends the operator to debug the wrong thing.
            # The message reaches the server log only (the API returns a fixed
            # string) and names the setting, never its value.
            raise InternalServiceFailure(
                "The blockchain data provider is not configured, so no real "
                f"transaction data can be acquired: {exc}"
            ) from exc
        except ProviderChainMismatch as exc:
            # The chain passed our own membership test but the provider still
            # refused it — in practice an ETHERSCAN_CHAIN_ID pinned to a
            # different chain than the one requested. That is a deployment
            # fault, so it must NOT come back as UNSUPPORTED_CHAIN: the caller
            # would retry with other chain names forever while the actual
            # problem sits in .env.
            raise InternalServiceFailure(
                "The blockchain data provider is misconfigured for chain "
                f"'{chain}': {exc}"
            ) from exc

        wallet_normalized = wallet.lower() if wallet else wallet

        if not provider.validate_address(wallet_normalized):
            await self._safe_close(provider)
            raise InvalidWalletError(
                f"'{wallet}' is not a valid address for chain '{chain}'."
            )

        # 2 & 3. Obtain the graph: provider -> normalize_all -> build_graph.
        # All three streams, with per-stream completeness recorded rather than
        # a failure being swallowed into a shorter list.
        try:
            acquisition = await acquire_wallet_transactions(
                provider,
                wallet_normalized,
                max_records_per_stream=self._settings.max_transactions_per_investigation,
            )
        except InvalidAddressError as exc:
            raise InvalidWalletError(str(exc)) from exc
        except IngestionError as exc:
            # The very first request of a stream failed, so nothing at all was
            # read. Reporting this as "no activity" would be a fabricated
            # negative; it is a service failure and says so.
            raise InternalServiceFailure(
                "The blockchain data provider could not be reached, so no "
                "transaction data could be acquired for this wallet."
            ) from exc
        finally:
            await self._safe_close(provider)

        if acquisition.total_records == 0:
            raise GraphNotFoundError(
                "GRAPH_NOT_FOUND: the provider returned no transaction "
                f"activity for '{wallet_normalized}' on chain '{chain}' — "
                "there is no data to build a graph from."
            )

        normalized = normalize_all(
            acquisition.native.records,
            acquisition.token.records,
            chain,
            acquisition.provider,
            internal_raw=acquisition.internal.records,
        )
        graph, _stats = build_graph(normalized)

        investigation_id = uuid.uuid4().hex

        graphs_dir = Path(self._settings.graph_cache_dir)
        graphs_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graphs_dir / f"{wallet_normalized}_{chain}_{investigation_id}.gpickle"
        save_graph(graph, graph_path)

        # 4. Execute the rest of the M1-M5 pipeline, unmodified.
        trace_result = trace_fund_flow(
            graph,
            wallet_normalized,
            settings=self._settings,
            max_hops=max_hops,
            max_paths=max_paths,
        )
        behavior_patterns = analyze_wallet_behavior(
            graph, wallet_normalized, self._settings, paths=trace_result.paths
        )

        seed_path = (
            self._settings.vasp_demo_seed_dataset_path
            if use_demo_seed
            else self._settings.vasp_seed_dataset_path
        )
        try:
            seed_entries = load_vasp_seed(seed_path)
        except SeedDataError as exc:
            raise InternalServiceFailure(
                "Could not load the VASP seed dataset required for attribution."
            ) from exc
        seed_index = build_seed_index(seed_entries)

        # 4 (cont). Produce the M4 attribution result.
        attribution = generate_candidates(trace_result, seed_index, behavior_patterns)

        # 5. Produce the M5 ML prediction.
        features = extract_wallet_features(
            graph, wallet_normalized, trace_result, behavior_patterns, attribution
        )
        model = train_model(seed=ml_seed if ml_seed is not None else DEFAULT_SEED)
        ml_prediction = predict(model, features)

        # 6-8. Persist the investigation, and M4/M5 results, separately.
        record = self._repository.create_investigation(
            investigation_id=investigation_id,
            wallet=wallet_normalized,
            chain=chain,
            max_hops=trace_result.max_hops,
            max_paths=trace_result.max_paths,
            graph_path=str(graph_path),
            search_truncated=attribution.search_truncated,
            attribution_status=attribution.status.value,
            ml_predicted_label=ml_prediction.predicted_label.value,
            training_data_type=ml_prediction.training_data_type,
        )
        self._repository.create_attribution(
            investigation_id=investigation_id, attribution_result=attribution
        )
        self._repository.create_ml_prediction(
            investigation_id=investigation_id, ml_prediction=ml_prediction
        )

        # 9. Return a structured investigation result.
        return InvestigationRunResult(
            investigation_id=investigation_id,
            wallet=wallet_normalized,
            chain=chain,
            max_hops=trace_result.max_hops,
            max_paths=trace_result.max_paths,
            graph_path=str(graph_path),
            attribution=attribution,
            ml_prediction=ml_prediction,
            record=record,
            data_complete=acquisition.complete,
            incompleteness_reasons=acquisition.incompleteness_reasons,
        )

    @staticmethod
    async def _safe_close(provider: BlockchainProvider) -> None:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()
