import json
import unittest
from urllib.error import URLError
from unittest.mock import patch

from providers import ProviderEngine, ProviderError


class _FakeResponse:
    def __init__(self, value):
        self._body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size=-1):
        return self._body[:size]


class ProviderEngineTests(unittest.TestCase):
    def test_ollama_uses_detected_installed_model(self):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/api/tags"):
                return _FakeResponse(
                    {"models": [{"name": "qwen2.5:latest"}, {"name": "llama3.2"}]}
                )
            return _FakeResponse({"message": {"content": "local answer"}})

        engine = ProviderEngine(
            fallback_order=("ollama",),
            environ={"OLLAMA_HOST": "http://localhost:11434"},
        )
        with patch("providers.urlopen", side_effect=fake_urlopen):
            result = engine.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "qwen2.5:latest")
        self.assertEqual(result.text, "local answer")
        chat_payload = json.loads(requests[1].data.decode("utf-8"))
        self.assertEqual(chat_payload["model"], "qwen2.5:latest")
        self.assertNotIn("format", chat_payload)

    def test_ollama_sends_structured_output_schema_and_zero_temperature(self):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/api/tags"):
                return _FakeResponse({"models": [{"name": "qwen2.5-coder:1.5b"}]})
            return _FakeResponse({"message": {"content": '{"type":"final"}'}})

        engine = ProviderEngine(
            fallback_order=("ollama",),
            environ={"OLLAMA_HOST": "http://localhost:11434"},
        )
        schema = {
            "type": "object",
            "properties": {"type": {"type": "string"}},
            "required": ["type"],
        }

        with patch("providers.urlopen", side_effect=fake_urlopen):
            engine.generate(
                [{"role": "user", "content": "hi"}],
                response_schema=schema,
            )

        payload = json.loads(requests[1].data.decode("utf-8"))
        self.assertEqual(payload["format"], schema)
        self.assertEqual(payload["options"], {"temperature": 0})

    def test_falls_back_to_openrouter_when_ollama_is_unreachable(self):
        def fake_urlopen(request, timeout):
            if request.full_url.startswith("http://localhost"):
                raise URLError("offline")
            return _FakeResponse(
                {"choices": [{"message": {"content": "cloud answer"}}]}
            )

        engine = ProviderEngine(
            environ={
                "OLLAMA_HOST": "http://localhost:11434",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_MODEL": "test/model",
            }
        )
        with patch("providers.urlopen", side_effect=fake_urlopen):
            result = engine.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.text, "cloud answer")

    def test_gemini_transport_error_does_not_expose_api_key(self):
        secret = "test-gemini-secret"
        engine = ProviderEngine(
            fallback_order=("gemini",),
            environ={"GEMINI_API_KEY": secret, "GEMINI_MODEL": "test-model"},
        )
        with patch("providers.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(ProviderError) as raised:
                engine.generate([{"role": "user", "content": "hello"}])

        self.assertNotIn(secret, str(raised.exception))

    def test_gemini_sends_inline_image_data(self):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return _FakeResponse(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "image described"}]}}
                    ]
                }
            )

        engine = ProviderEngine(
            fallback_order=("gemini",),
            environ={"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"},
        )
        message = {
            "role": "user",
            "content": "Describe this",
            "images": [{"mime_type": "image/png", "base64_data": "aGVsbG8="}],
        }
        with patch("providers.urlopen", side_effect=fake_urlopen):
            result = engine.generate([message])

        payload = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(
            payload["contents"][0]["parts"][1],
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": "aGVsbG8=",
                }
            },
        )
        self.assertEqual(result.text, "image described")

    def test_invalid_messages_are_rejected_before_provider_access(self):
        engine = ProviderEngine(
            fallback_order=("openrouter",),
            environ={},
        )
        with patch("providers.urlopen") as mock_urlopen:
            with self.assertRaises(ValueError):
                engine.generate([{"role": "tool", "content": "not supported"}])
        mock_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
