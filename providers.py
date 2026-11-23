"""Provider selection and chat completion adapters for AutonomousAgent.

Configuration is read from the process environment. This module does not load
or parse dotenv files.
"""

from __future__ import annotations

import json
import os
import socket
from base64 import b64decode
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_FALLBACK_ORDER = ("ollama", "openrouter", "gemini")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_VALID_ROLES = {"system", "user", "assistant"}


class ProviderError(RuntimeError):
    """Raised when a provider cannot complete a request."""


@dataclass(frozen=True)
class ImageInput:
    mime_type: str
    base64_data: str


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    images: Tuple[ImageInput, ...] = ()


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    model: str
    text: str


def _normalize_messages(
    messages: Sequence[ChatMessage | Mapping[str, object]],
) -> List[Dict[str, object]]:
    if not messages:
        raise ValueError("At least one chat message is required")

    normalized: List[Dict[str, object]] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            role, content = message.role, message.content
            images = message.images
        elif isinstance(message, Mapping):
            role, content = message.get("role"), message.get("content")
            images = message.get("images", ())
        else:
            raise TypeError("Messages must be ChatMessage objects or mappings")

        if not isinstance(role, str) or role not in _VALID_ROLES:
            raise ValueError("Message role must be system, user, or assistant")
        if not isinstance(content, str):
            raise TypeError("Message content must be a string")
        if not isinstance(images, (list, tuple)):
            raise TypeError("Message images must be a list or tuple")
        if images and role != "user":
            raise ValueError("Images may only be attached to user messages")

        normalized_images = []
        for image in images:
            if isinstance(image, ImageInput):
                mime_type, base64_data = image.mime_type, image.base64_data
            elif isinstance(image, Mapping):
                mime_type = image.get("mime_type")
                base64_data = image.get("base64_data")
            else:
                raise TypeError("Images must be ImageInput objects or mappings")
            if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
                raise ValueError("Image mime_type must be an image type")
            if not isinstance(base64_data, str):
                raise TypeError("Image base64_data must be a string")
            try:
                b64decode(base64_data, validate=True)
            except (ValueError, TypeError):
                raise ValueError("Image base64_data is invalid") from None
            normalized_images.append(
                {"mime_type": mime_type, "base64_data": base64_data}
            )
        normalized.append(
            {"role": role, "content": content, "images": normalized_images}
        )

    return normalized


def _request_json(
    provider: str,
    url: str,
    *,
    payload: Optional[Mapping[str, object]] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float,
) -> Dict[str, object]:
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    try:
        request = Request(url, data=data, headers=request_headers)
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise ProviderError(
            "{} returned HTTP status {}".format(provider, exc.code)
        ) from None
    except (URLError, TimeoutError, socket.timeout, OSError, ValueError) as exc:
        raise ProviderError(
            "{} request failed ({})".format(provider, type(exc).__name__)
        ) from None

    if len(raw_body) > _MAX_RESPONSE_BYTES:
        raise ProviderError("{} response exceeded the size limit".format(provider))
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProviderError("{} returned invalid JSON".format(provider)) from None
    if not isinstance(decoded, dict):
        raise ProviderError("{} returned an unexpected response".format(provider))
    return decoded


class ProviderEngine:
    """Select providers in order, falling back when one cannot respond."""

    def __init__(
        self,
        fallback_order: Optional[Sequence[str]] = None,
        timeout: float = 60.0,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        order = tuple(
            DEFAULT_FALLBACK_ORDER if fallback_order is None else fallback_order
        )
        valid_providers = {"ollama", "openrouter", "gemini"}
        if not order or len(set(order)) != len(order):
            raise ValueError("Fallback order must contain unique providers")
        if any(provider not in valid_providers for provider in order):
            raise ValueError("Fallback order contains an unknown provider")
        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero")

        self.fallback_order = order
        self.timeout = timeout
        self._environ = dict(os.environ if environ is None else environ)
        self._ollama_url = self._normalize_ollama_url(
            self._environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
        )

    @staticmethod
    def _normalize_ollama_url(url: str) -> str:
        normalized = url.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_HOST must use http:// or https://")
        return normalized

    def list_ollama_models(self) -> Tuple[str, ...]:
        """Return model names currently installed on the configured Ollama host."""
        response = _request_json(
            "ollama",
            self._ollama_url + "/api/tags",
            timeout=self.timeout,
        )
        models = response.get("models")
        if not isinstance(models, list):
            raise ProviderError("ollama returned an unexpected model list")

        names = []
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                if model["name"] and model["name"] not in names:
                    names.append(model["name"])
        return tuple(names)

    def generate(
        self,
        messages: Sequence[ChatMessage | Mapping[str, object]],
        response_schema: Optional[Mapping[str, object]] = None,
    ) -> ProviderResponse:
        normalized = _normalize_messages(messages)
        failures = []
        for provider in self.fallback_order:
            try:
                if provider == "ollama":
                    return self._generate_ollama(normalized, response_schema)
                if provider == "openrouter":
                    return self._generate_openrouter(normalized)
                return self._generate_gemini(normalized)
            except ProviderError as exc:
                failures.append("{}: {}".format(provider, exc))

        raise ProviderError("All providers failed: " + "; ".join(failures))

    def _generate_ollama(
        self,
        messages: Sequence[Mapping[str, object]],
        response_schema: Optional[Mapping[str, object]] = None,
    ) -> ProviderResponse:
        models = self.list_ollama_models()
        configured_model = self._environ.get("OLLAMA_MODEL")
        if configured_model:
            if configured_model not in models:
                raise ProviderError(
                    "OLLAMA_MODEL is not installed on the configured Ollama host"
                )
            model = configured_model
        elif models:
            model = models[0]
        else:
            raise ProviderError("no installed Ollama models were detected")

        payload: Dict[str, object] = {
            "model": model,
            "messages": [
                {
                    "role": message["role"],
                    "content": message["content"],
                    "images": [
                        image["base64_data"] for image in message["images"]
                    ],
                }
                for message in messages
            ],
            "stream": False,
        }
        if response_schema is not None:
            payload["format"] = response_schema
            payload["options"] = {"temperature": 0}

        response = _request_json(
            "ollama",
            self._ollama_url + "/api/chat",
            payload=payload,
            timeout=self.timeout,
        )
        message = response.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise ProviderError("ollama returned no chat content")
        return ProviderResponse("ollama", model, text)

    def _generate_openrouter(
        self, messages: Sequence[Mapping[str, object]]
    ) -> ProviderResponse:
        api_key = self._environ.get("OPENROUTER_API_KEY")
        model = self._environ.get("OPENROUTER_MODEL")
        if not api_key:
            raise ProviderError("OPENROUTER_API_KEY is not configured")
        if not model:
            raise ProviderError("OPENROUTER_MODEL is not configured")

        headers = {"Authorization": "Bearer " + api_key}
        site_url = self._environ.get("OPENROUTER_SITE_URL")
        app_name = self._environ.get("OPENROUTER_APP_NAME")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        openrouter_messages = []
        for message in messages:
            images = message["images"]
            if images:
                content = []
                if message["content"]:
                    content.append({"type": "text", "text": message["content"]})
                for image in images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:{};base64,{}".format(
                                    image["mime_type"], image["base64_data"]
                                )
                            },
                        }
                    )
            else:
                content = message["content"]
            openrouter_messages.append(
                {"role": message["role"], "content": content}
            )

        response = _request_json(
            "openrouter",
            "https://openrouter.ai/api/v1/chat/completions",
            payload={
                "model": model,
                "messages": openrouter_messages,
                "stream": False,
            },
            headers=headers,
            timeout=self.timeout,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("openrouter returned no chat choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise ProviderError("openrouter returned no chat content")
        return ProviderResponse("openrouter", model, text)

    def _generate_gemini(
        self, messages: Sequence[Mapping[str, object]]
    ) -> ProviderResponse:
        api_key = self._environ.get("GEMINI_API_KEY")
        model = self._environ.get("GEMINI_MODEL")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not configured")
        if not model:
            raise ProviderError("GEMINI_MODEL is not configured")

        system_parts = [
            message["content"]
            for message in messages
            if message["role"] == "system" and message["content"]
        ]
        contents = []
        for message in messages:
            if message["role"] == "system":
                if message["images"]:
                    raise ProviderError("gemini does not support system images")
                continue
            parts = []
            if message["content"]:
                parts.append({"text": message["content"]})
            parts.extend(
                {
                    "inline_data": {
                        "mime_type": image["mime_type"],
                        "data": image["base64_data"],
                    }
                }
                for image in message["images"]
            )
            contents.append(
                {
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": parts,
                }
            )
        if not contents:
            raise ProviderError("gemini requires at least one user or assistant message")

        payload: Dict[str, object] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n".join(system_parts)}]
            }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + quote(model, safe="")
            + ":generateContent?"
            + urlencode({"key": api_key})
        )
        response = _request_json(
            "gemini",
            url,
            payload=payload,
            timeout=self.timeout,
        )

        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderError("gemini returned no chat candidates")
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise ProviderError("gemini returned no chat content")
        text = "".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if not text:
            raise ProviderError("gemini returned no text content")
        return ProviderResponse("gemini", model, text)