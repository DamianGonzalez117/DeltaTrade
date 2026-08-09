"""TDD — Tests for BitsoAuth (RED phase).

Bitso API v3 authentication:
  Headers:
    Authorization: Bitso <api_key>
    X-Request-Timestamp: <nonce>
    X-Request-Signature: <hex HMAC-SHA256>

  Signature = HMAC-SHA256(secret, nonce + METHOD + path + body)
"""
import hashlib
import hmac
import time

import pytest

from trading_system.connectors.bitso.auth import BitsoAuth


def _expected_signature(secret: str, message: str) -> str:
    return hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


class TestBitsoAuthHeaderFormat:
    def test_headers_contain_authorization_with_bitso_prefix(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("GET", "/v3/balance/")
        assert headers["Authorization"] == f"Bitso {api_key}"

    def test_headers_contain_timestamp(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("GET", "/v3/balance/")
        assert "X-Request-Timestamp" in headers
        nonce = headers["X-Request-Timestamp"]
        assert nonce.isdigit()
        assert len(nonce) >= 10

    def test_headers_contain_signature(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("GET", "/v3/balance/")
        sig = headers["X-Request-Signature"]
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


class TestBitsoAuthSignatureCorrectness:
    def test_get_request_no_body(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("GET", "/v3/balance/")
        nonce = headers["X-Request-Timestamp"]
        message = nonce + "GET" + "/v3/balance/"
        assert headers["X-Request-Signature"] == _expected_signature(api_secret, message)

    def test_post_request_with_body(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        body = '{"book":"btc_mxn","side":"buy","type":"market","major":"0.01"}'
        headers = auth.build_headers("POST", "/v3/orders/", body)
        nonce = headers["X-Request-Timestamp"]
        message = nonce + "POST" + "/v3/orders/" + body
        assert headers["X-Request-Signature"] == _expected_signature(api_secret, message)

    def test_delete_request(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers = auth.build_headers("DELETE", "/v3/orders/abc123")
        nonce = headers["X-Request-Timestamp"]
        message = nonce + "DELETE" + "/v3/orders/abc123"
        assert headers["X-Request-Signature"] == _expected_signature(api_secret, message)


class TestBitsoAuthMethodCaseInsensitivity:
    def test_method_uppercased_in_signature(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        headers_lower = auth.build_headers("post", "/v3/orders/", "{}")
        headers_upper = auth.build_headers("POST", "/v3/orders/", "{}")
        assert headers_lower["X-Request-Signature"] == headers_upper["X-Request-Signature"]


class TestBitsoAuthNonceFreshness:
    def test_different_calls_produce_different_nonces(self, api_key, api_secret):
        auth = BitsoAuth(api_key, api_secret)
        h1 = auth.build_headers("GET", "/v3/balance/")
        time.sleep(0.01)
        h2 = auth.build_headers("GET", "/v3/balance/")
        assert h1["X-Request-Timestamp"] != h2["X-Request-Timestamp"]


class TestBitsoAuthEmptyCredentials:
    def test_empty_key_raises_value_error(self, api_secret):
        with pytest.raises(ValueError, match="API key"):
            BitsoAuth("", api_secret)

    def test_empty_secret_raises_value_error(self, api_key):
        with pytest.raises(ValueError, match="API secret"):
            BitsoAuth(api_key, "")