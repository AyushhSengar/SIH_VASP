from app.blockchain.etherscan import EtherscanProvider
from app.core.config import Settings


def make_provider() -> EtherscanProvider:
    settings = Settings(
        etherscan_api_key="test",
        etherscan_base_url="https://example.invalid",
        etherscan_chain_id=1,
        max_transactions_per_investigation=100,
        default_lookback_days=90,
        http_timeout_seconds=5,
        http_max_retries=1,
    )
    return EtherscanProvider(settings)


def test_valid_address():
    p = make_provider()
    assert p.validate_address("0x1234567890abcdef1234567890abcdef12345678") is True


def test_invalid_address_wrong_length():
    p = make_provider()
    assert p.validate_address("0x1234") is False


def test_invalid_address_missing_prefix():
    p = make_provider()
    assert p.validate_address("1234567890abcdef1234567890abcdef12345678") is False


def test_invalid_address_bad_chars():
    p = make_provider()
    assert p.validate_address("0x" + "z" * 40) is False


def test_empty_address():
    p = make_provider()
    assert p.validate_address("") is False
    assert p.validate_address(None) is False
