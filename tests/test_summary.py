from app.models import AssetType, NormalizedTransfer, TransferStatus
from app.normalization.summary import summarize_wallet

WALLET = "0xaaaa111111111111111111111111111111111a"
COUNTERPARTY_1 = "0xbbbb222222222222222222222222222222222b"
COUNTERPARTY_2 = "0xcccc333333333333333333333333333333333c"


def make_transfer(from_addr, to_addr, amount, ts, asset_type=AssetType.NATIVE):
    return NormalizedTransfer(
        tx_hash=f"0x{ts}",
        chain="ethereum",
        block_number=1,
        timestamp=ts,
        from_address=from_addr,
        to_address=to_addr,
        asset_type=asset_type,
        amount_raw=str(int(amount * 1e18)),
        amount=amount,
        status=TransferStatus.SUCCESS,
        source_provider="etherscan",
        fetched_at=0,
    )


def test_summarize_wallet_empty():
    s = summarize_wallet(WALLET, "ethereum", [])
    assert s.transaction_count == 0
    assert s.data_completeness_note is not None


def test_summarize_wallet_inflow_and_outflow():
    transfers = [
        make_transfer(COUNTERPARTY_1, WALLET, 2.0, 100),
        make_transfer(WALLET, COUNTERPARTY_2, 0.5, 200),
    ]
    s = summarize_wallet(WALLET, "ethereum", transfers)
    assert s.total_inflow_native == 2.0
    assert s.total_outflow_native == 0.5
    assert s.unique_senders == 1
    assert s.unique_receivers == 1
    assert s.first_seen == 100
    assert s.last_seen == 200


def test_summarize_wallet_ignores_token_transfers():
    transfers = [
        make_transfer(COUNTERPARTY_1, WALLET, 100.0, 100, asset_type=AssetType.ERC20),
    ]
    s = summarize_wallet(WALLET, "ethereum", transfers)
    assert s.total_inflow_native == 0.0  # token amounts excluded in Milestone 1
