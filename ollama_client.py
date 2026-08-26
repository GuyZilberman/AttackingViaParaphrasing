"""
Thin HTTP client for the Ollama REST API.
Only requires the `requests` standard-ish library (pre-installed on this VM).
"""

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

import requests

OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaError(Exception):
    pass


class OllamaClient:
    """
    Minimal wrapper around Ollama's /api/chat endpoint.
    All model calls go through `chat()`.  JSON-structured replies go through
    `chat_json()` which adds format="json" and parses the result.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Connectivity helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        r = self._session.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def require_available(self) -> None:
        """Raise a clear error if the Ollama server isn't reachable."""
        if not self.is_available():
            raise OllamaError(
                "Ollama server is not running. "
                "Start it with:  bash start_ollama.sh"
            )

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        think: bool = False,
    ) -> str:
        """
        Send a chat request and return the assistant's text content.

        Args:
            model:       Ollama model tag, e.g. "qwen3:4b".
            messages:    List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature (0 = greedy).
            max_tokens:  Maximum tokens to generate.
            think:       Enable chain-of-thought (qwen3 /think mode).
                         Set False to suppress <think> blocks for speed.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            r = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                "Cannot reach Ollama server. Run: bash start_ollama.sh"
            ) from exc

        if r.status_code != 200:
            raise OllamaError(
                f"Ollama API returned {r.status_code}: {r.text[:400]}"
            )

        data = r.json()
        content = data.get("message", {}).get("content", "")
        return content.strip()

    def chat_json(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        think: bool = False,
    ) -> Any:
        """
        Like `chat()` but instructs Ollama to return valid JSON and
        automatically parses it.  Falls back to regex extraction on failure.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "think": think,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            r = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                "Cannot reach Ollama server. Run: bash start_ollama.sh"
            ) from exc

        if r.status_code != 200:
            raise OllamaError(
                f"Ollama API returned {r.status_code}: {r.text[:400]}"
            )

        data = r.json()
        content = data.get("message", {}).get("content", "").strip()
        return _parse_json_robust(content)


# ------------------------------------------------------------------
# JSON parsing helpers (shared with evaluators)
# ------------------------------------------------------------------

def _balanced_json_substring(text: str) -> Optional[str]:
    opens = {"{": "}", "[": "]"}
    stack: List[str] = []
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch in opens:
            if not stack:
                start = i
            stack.append(opens[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                return text[start : i + 1]
    return None


def _parse_json_robust(text: str) -> Any:
    """
    Parse JSON from model output, tolerating surrounding prose.
    Raises ValueError if nothing parseable is found.
    """
    # 1. Try directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try the first balanced JSON block
    sub = _balanced_json_substring(text)
    if sub:
        try:
            return json.loads(sub)
        except json.JSONDecodeError:
            pass

    # 3. Strip markdown code fences and retry
    stripped = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    sub2 = _balanced_json_substring(stripped)
    if sub2:
        try:
            return json.loads(sub2)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model output: {text[:300]!r}")


# ------------------------------------------------------------------
# Text normalisation (reused by exact-match evaluator)
# ------------------------------------------------------------------

def normalise_text(s: str) -> str:
    """Lowercase, strip accents, keep only alphanumeric + spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()
