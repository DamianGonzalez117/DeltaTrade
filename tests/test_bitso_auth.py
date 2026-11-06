"""TDD — Tests for BitsoAuth.

Bitso API v3 authentication (docs.bitso.com, esquema vigente):
  UN solo header:
    Authorization: Bitso <api_key>:<nonce>:<hex HMAC-SHA256>

  Signature = HMAC-SHA256(secret, nonce + METHOD + path + body)
  Nonce v2 (obligatorio desde nov-2025) = timestamp ms (13 dígitos) +
  salt aleatorio (1-6 dígitos; aquí 6) → 19 dígitos.
"""
import hashlib
import hmac
import re
import time
from unittest.mock import patch

import pytest

from trading_system.connectors.bitso.auth import BitsoAuth

_HEADER_RE = re.compile(r"^Bitso (?P<key>[^:]+):(?P<nonce>\d{14,19}):(?P<sig>[0-9a-f]{64})$")


def _expected_signature(secret: str, message: str) -> str:
    return hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


def _parse_header(headers: dict[str, str]) -> dict[str, str]:
    match = _HEADER_RE.match(headers["Authorization"])
    assert match is not None, f"Header malformado: {headers.get('Authorization')}"
    return match.groupdict()


class TestBitsoAuthHeaderFormat:
    def test_single_authorization_header_with_key_nonce_signature(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("GET", "/v3/balance/")

        assert set(headers) == {"Authorization"}
        parts = _parse_header(headers)
        assert parts["key"] == api_key

    def test_legacy_split_headers_not_sent(self, api_key, api_secret):
        """El esquema viejo de 3 headers ya no aplica (causaba 400)."""
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("GET", "/v3/balance/")

        assert "X-Request-Timestamp" not in headers
        assert "X-Request-Signature" not in headers

    def test_nonce_v2_format(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("GET", "/v3/balance/")
        nonce = _parse_header(headers)["nonce"]

        assert nonce.isdigit()
        assert 14 <= len(nonce) <= 19
        # los primeros 13 dígitos son un timestamp en ms reciente
        timestamp_ms = int(nonce[:13])
        assert abs(timestamp_ms - int(time.time() * 1000)) < 60_000


class TestBitsoAuthSignatureCorrectness:
    def test_get_request_no_body(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("GET", "/v3/balance/")
        parts = _parse_header(headers)
        message = parts["nonce"] + "GET" + "/v3/balance/"
        assert parts["sig"] == _expected_signature(api_secret, message)

    def test_post_request_with_body(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        body = '{"book":"btc_mxn","side":"buy","type":"market","major":"0.01"}'
        headers = auth.build_headers("POST", "/v3/orders/", body)
        parts = _parse_header(headers)
        message = parts["nonce"] + "POST" + "/v3/orders/" + body
        assert parts["sig"] == _expected_signature(api_secret, message)

    def test_delete_request(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("DELETE", "/v3/orders/abc123")
        parts = _parse_header(headers)
        message = parts["nonce"] + "DELETE" + "/v3/orders/abc123"
        assert parts["sig"] == _expected_signature(api_secret, message)


class TestBitsoAuthMethodCaseInsensitivity:
    def test_method_uppercased_in_signature(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        with patch(
            "trading_system.connectors.bitso.auth.random.randint",
            return_value=123456,
        ):
            headers_lower = auth.build_headers("post", "/v3/orders/", "{}")
            headers_upper = auth.build_headers("POST", "/v3/orders/", "{}")
        assert headers_lower["Authorization"] == headers_upper["Authorization"]


class TestBitsoAuthNonceFreshness:
    def test_different_calls_produce_different_nonces(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        h1 = auth.build_headers("GET", "/v3/balance/")
        time.sleep(0.01)
        h2 = auth.build_headers("GET", "/v3/balance/")

        nonce1 = _parse_header(h1)["nonce"]
        nonce2 = _parse_header(h2)["nonce"]
        assert nonce1 != nonce2


class TestBitsoAuthEmptyCredentials:
    def test_empty_key_raises_value_error(self, api_secret):
        with pytest.raises(ValueError, match="API key"):
            BitsoAuth("", api_secret)

    def test_empty_secret_raises_value_error(self, api_key):
        with pytest.raises(ValueError, match="API secret"):
            BitsoAuth(api_key, "")
