from app.models import AssetType, TransferStatus
from app.normalization.transactions import (
    normalize_all,
    normalize_native_transaction,
    normalize_token_transfer,
)

RAW_NATIVE_TX = {
    "hash": "0xabc123",
    "blockNumber": "18000000",
    "timeStamp": "1700000000",
    "from": "0xAAAA111111111111111111111111111111111a",
    "to": "0xBBBB222222222222222222222222222222222b",
    "value": "1000000000000000000",  # 1 ETH
    "isError": "0",
    "gasUsed": "21000",
    "gasPrice": "20000000000",
    "methodId": "0x",
}

RAW_FAILED_TX = {**RAW_NATIVE_TX, "hash": "0xdef456", "isError": "1"}

RAW_CONTRACT_CREATION_TX = {**RAW_NATIVE_TX, "hash": "0xghi789", "to": ""}

RAW_TOKEN_TX = {
    "hash": "0xtoken001",
    "blockNumber": "18000001",
    "timeStamp": "1700000100",
    "from": "0xCCCC333333333333333333333333333333333c",
    "to": "0xDDDD444444444444444444444444444444444d",
    "value": "5000000",  # 5 USDC (6 decimals)
    "tokenDecimal": "6",
    "tokenSymbol": "USDC",
    "contractAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "gasUsed": "65000",
}


def test_normalize_native_transaction_basic_fields():
    t = normalize_native_transaction(RAW_NATIVE_TX, "ethereum", "etherscan")
    assert t.tx_hash == "0xabc123"
    assert t.chain == "ethereum"
    assert t.block_number == 18000000
    assert t.timestamp == 1700000000
    assert t.from_address == "0xaaaa111111111111111111111111111111111a"
    assert t.to_address == "0xbbbb222222222222222222222222222222222b"
    assert t.asset_type == AssetType.NATIVE
    assert t.amount == 1.0
    assert t.status == TransferStatus.SUCCESS
    assert t.is_contract_creation is False
    assert t.source_provider == "etherscan"


def test_normalize_native_transaction_gas_fee_calculation():
    t = normalize_native_transaction(RAW_NATIVE_TX, "ethereum", "etherscan")
    # 21000 gas * 20 gwei = 0.00042 ETH
    assert round(t.gas_fee_native, 8) == 0.00042


def test_normalize_failed_transaction_marks_status():
    t = normalize_native_transaction(RAW_FAILED_TX, "ethereum", "etherscan")
    assert t.status == TransferStatus.FAILED


def test_normalize_contract_creation_has_no_to_address():
    t = normalize_native_transaction(RAW_CONTRACT_CREATION_TX, "ethereum", "etherscan")
    assert t.to_address is None
    assert t.is_contract_creation is True


def test_normalize_token_transfer_applies_decimals():
    t = normalize_token_transfer(RAW_TOKEN_TX, "ethereum", "etherscan")
    assert t.asset_type == AssetType.ERC20
    assert t.asset_symbol == "USDC"
    assert t.amount == 5.0  # 5,000,000 / 10^6
    assert t.asset_identifier == "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


def test_normalize_all_merges_and_sorts_by_timestamp():
    later_native = {**RAW_NATIVE_TX, "hash": "0xzzz", "timeStamp": "1700000200"}
    merged = normalize_all(
        native_raw=[later_native, RAW_NATIVE_TX],
        token_raw=[RAW_TOKEN_TX],
        chain="ethereum",
        source_provider="etherscan",
    )
    assert len(merged) == 3
    timestamps = [t.timestamp for t in merged]
    assert timestamps == sorted(timestamps)


def test_normalize_handles_missing_optional_fields_gracefully():
    sparse = {"hash": "0xsparse", "from": "0xAAAA", "value": "0"}
    t = normalize_native_transaction(sparse, "ethereum", "etherscan")
    assert t.tx_hash == "0xsparse"
    assert t.amount == 0.0
    assert t.gas_fee_native is None
    assert t.block_number == 0
