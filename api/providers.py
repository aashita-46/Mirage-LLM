"""Live provider adapters. Secrets never enter returned metadata."""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from api.core import (
    DatasetExample,
    GenerationRecord,
    ModelInfo,
    ProviderCapabilities,
    SamplingConfig,
)

DEFAULT_TIMEOUT = float(os.getenv("MIRAGE_PROVIDER_TIMEOUT", "120"))
SELF_VERIFICATION_PROMPT_VERSION = "mirage-self-verification-v1"


def redact(value: str) -> str:
    """Remove known credentials from provider error strings."""
    secrets = [
        os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
    ]
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


class ProviderError(RuntimeError):
    def __init__(self, message: str, code: str = "provider_error"):
        super().__init__(redact(message))
        self.code = code


class BaseProvider(ABC):
    info: ModelInfo

    @abstractmethod
    def generate(self, example: DatasetExample, config: SamplingConfig) -> GenerationRecord:
        raise NotImplementedError


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)
        self.info = ModelInfo(
            provider="ollama", model=model, mode="live",
            capabilities=ProviderCapabilities(
                supports_logprobs=False, supports_seed=True, supports_streaming=True,
                supports_parallel_samples=False, supports_token_usage=True,
                supports_structured_output=True, supports_retrieval=False, supports_vision=False,
            ),
        )

    def available_models(self) -> list[str]:
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return [item["name"] for item in response.json().get("models", [])]
        except Exception as exc:
            raise ProviderError(f"Ollama is unavailable: {exc}", "provider_unreachable") from exc

    def validate(self) -> None:
        models = self.available_models()
        if self.model not in models:
            raise ProviderError(f"Ollama model {self.model!r} is not installed. Available: {', '.join(models) or 'none'}", "model_not_found")

    def _chat(self, messages: list[dict[str, str]], config: SamplingConfig, *, json_output: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {
                "temperature": config.temperature, "top_p": config.top_p,
                "num_predict": config.max_tokens, "seed": config.seed,
            },
        }
        if json_output:
            payload["format"] = "json"
        response = self.client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    def generate(self, example: DatasetExample, config: SamplingConfig) -> GenerationRecord:
        started = time.perf_counter()
        errors: list[str] = []
        outputs: list[str] = []
        usage = {"input": 0, "output": 0}
        try:
            self.validate()
            messages = [
                {"role": "system", "content": "Answer concisely. Reject false premises and say when information is insufficient."},
                {"role": "user", "content": example.question},
            ]
            for index in range(config.semantic_samples):
                try:
                    sample_config = config.model_copy(update={"seed": config.seed + index})
                    raw = self._chat(messages, sample_config)
                    outputs.append(raw.get("message", {}).get("content", "").strip())
                    usage["input"] += int(raw.get("prompt_eval_count", 0))
                    usage["output"] += int(raw.get("eval_count", 0))
                except Exception as exc:
                    errors.append(redact(f"sample {index + 1}: {exc}"))
            if not outputs:
                raise ProviderError("; ".join(errors) or "Ollama returned no responses.", "invalid_response")
            verification = None
            try:
                verify_prompt = (
                    "Independently check the answer from scratch; do not assume or repeat its conclusion. Return JSON with verdict "
                    "(supported|uncertain|contradicted), confidence from 0 to 1, reason, "
                    f"and claims. Question: {example.question}\nAnswer: {outputs[0]}"
                )
                raw_verify = self._chat([{"role": "user", "content": verify_prompt}], config, json_output=True)
                verification = json.loads(raw_verify.get("message", {}).get("content", "{}"))
            except Exception as exc:
                errors.append(redact(f"self-verification: {exc}"))
            return GenerationRecord(
                response=outputs[0], sampled_responses=outputs, token_usage=usage,
                latency_ms=(time.perf_counter() - started) * 1000,
                provider_metadata={
                    "execution_mode": "live", "provider": "ollama", "model": self.model,
                    "sampling": config.model_dump(), "partial_errors": errors,
                    "self_verification": verification, "self_verification_raw": raw_verify.get("message", {}).get("content") if verification is not None else None,
                    "self_verification_prompt_version": SELF_VERIFICATION_PROMPT_VERSION,
                    "self_verification_temperature": config.temperature,
                    "self_verification_max_tokens": config.max_tokens, "retry_count": 0,
                },
            )
        except Exception as exc:
            return GenerationRecord(
                response="", sampled_responses=outputs, token_usage=usage,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=redact(str(exc)),
                provider_metadata={
                    "execution_mode": "live", "provider": "ollama", "model": self.model,
                    "error_code": exc.code if isinstance(exc, ProviderError) else ("timeout" if isinstance(exc, httpx.TimeoutException) else "provider_error"),
                },
            )


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = 2,
        supports_logprobs: bool | None = None,
        client: httpx.Client | None = None,
    ):
        self.model = model or os.getenv("OPENAI_COMPATIBLE_MODEL", "")
        self.base_url = (base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
        self.timeout = timeout
        self.retries = retries
        configured_logprobs = os.getenv("OPENAI_COMPATIBLE_SUPPORTS_LOGPROBS", "false").lower() == "true"
        self.supports_logprobs = configured_logprobs if supports_logprobs is None else supports_logprobs
        self.client = client or httpx.Client(timeout=timeout)
        self.info = ModelInfo(
            provider="openai_compatible", model=self.model or "unconfigured", mode="live",
            capabilities=ProviderCapabilities(
                supports_logprobs=self.supports_logprobs, supports_seed=True,
                supports_streaming=True, supports_parallel_samples=True,
                supports_token_usage=True, supports_structured_output=True,
                supports_retrieval=False, supports_vision=False,
            ),
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url or not self.model:
            raise ProviderError("OpenAI-compatible base URL and model must be configured.", "invalid_configuration")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                retryable = status is None or status in {408, 429, 500, 502, 503, 504}
                if attempt < self.retries and retryable:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                break
        code = "timeout" if isinstance(last_error, httpx.TimeoutException) else (
            "rate_limited" if isinstance(last_error, httpx.HTTPStatusError) and last_error.response.status_code == 429 else
            "authentication_failed" if isinstance(last_error, httpx.HTTPStatusError) and last_error.response.status_code in {401, 403} else
            "provider_error"
        )
        raise ProviderError(f"OpenAI-compatible request failed: {last_error}", code)

    def generate(self, example: DatasetExample, config: SamplingConfig) -> GenerationRecord:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": example.question}],
            "temperature": config.temperature, "top_p": config.top_p,
            "max_tokens": config.max_tokens, "seed": config.seed,
            "n": config.semantic_samples,
        }
        if self.supports_logprobs:
            payload.update({"logprobs": True, "top_logprobs": 5})
        try:
            raw = self._request(payload)
            choices = raw.get("choices", [])
            outputs = [choice.get("message", {}).get("content", "").strip() for choice in choices]
            outputs = [item for item in outputs if item]
            if not outputs:
                raise ProviderError("OpenAI-compatible endpoint returned no usable choices.")
            token_logprobs = None
            if self.supports_logprobs and choices:
                content = (choices[0].get("logprobs") or {}).get("content") or []
                token_logprobs = [float(item["logprob"]) for item in content if item.get("logprob") is not None] or None
            usage = raw.get("usage") or {}
            return GenerationRecord(
                response=outputs[0], sampled_responses=outputs,
                token_logprobs=token_logprobs,
                token_usage={
                    "input": int(usage.get("prompt_tokens", 0)),
                    "output": int(usage.get("completion_tokens", 0)),
                },
                latency_ms=(time.perf_counter() - started) * 1000,
                provider_metadata={
                    "execution_mode": "live", "provider": "openai_compatible",
                    "model": self.model, "sampling": config.model_dump(),
                    "retry_limit": self.retries,
                },
            )
        except Exception as exc:
            return GenerationRecord(
                response="", sampled_responses=[], latency_ms=(time.perf_counter() - started) * 1000,
                provider_metadata={
                    "execution_mode": "live", "provider": "openai_compatible", "model": self.model,
                    "error_code": exc.code if isinstance(exc, ProviderError) else ("timeout" if isinstance(exc, httpx.TimeoutException) else "provider_error"),
                },
                error=redact(str(exc)),
            )


def provider_for(name: str, model: str) -> BaseProvider:
    if name == "ollama":
        return OllamaProvider(model)
    if name == "openai_compatible":
        return OpenAICompatibleProvider(model=model)
    from api.core import CachedDemoProvider
    if name == "cached_demo":
        return CachedDemoProvider()
    raise ProviderError(f"Unknown provider: {name}")
