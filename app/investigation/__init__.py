"""
MACRO MILESTONE 6 — investigation orchestration service.

This package contains NO blockchain-intelligence logic of its own. It
only calls, in order, the existing M1–M5 functions:

    EtherscanProvider -> normalize_all -> build_graph -> trace_fund_flow
    -> analyze_wallet_behavior -> generate_candidates
    -> extract_wallet_features -> train_model -> predict

and hands the results to app/db for persistence.
"""
