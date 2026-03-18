# SPDX-License-Identifier: Apache-2.0
"""
Outlines-based JSON Schema logits processor for oMLX.

Provides token-level constrained decoding that guarantees valid JSON output
matching a given JSON Schema. Uses Outlines' JSONLogitsProcessor under the
hood, wrapped to be compatible with mlx-lm's (tokens, logits) -> logits
interface used by oMLX's scheduler.

This follows the same pattern as ThinkingBudgetProcessor in thinking.py.

Requires: pip install "outlines>=1.0.0"
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("omlx.api.json_logits_processor")

# Cache compiled processors per schema to avoid re-compilation overhead.
# Outlines compiles JSON schema -> regex -> state machine on first use,
# which can take 1-5 seconds. Caching makes subsequent requests instant.
_processor_cache: Dict[str, Any] = {}
_MAX_CACHE_SIZE = 32


def _schema_cache_key(schema: dict) -> str:
    """Create a stable cache key from a JSON schema dict."""
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def _get_or_create_outlines_processor(schema: dict, tokenizer: Any) -> Any:
    """Get a cached Outlines JSONLogitsProcessor or create a new one.

    Args:
        schema: JSON Schema dict (the "schema" field from response_format).
        tokenizer: The model's tokenizer (from mlx-lm).

    Returns:
        An Outlines JSONLogitsProcessor instance.

    Raises:
        ImportError: If outlines is not installed.
        ValueError: If the schema is invalid.
    """
    try:
        from outlines.processors.structured import JSONLogitsProcessor
        from outlines.models import TransformerTokenizer
    except ImportError:
        raise ImportError(
            "Outlines is required for token-level JSON schema enforcement. "
            "Install it with: pip install 'outlines>=1.0.0,<1.2.0'\n"
            "Without Outlines, oMLX falls back to prompt-based JSON guidance."
        )

    cache_key = _schema_cache_key(schema)

    if cache_key in _processor_cache:
        return _processor_cache[cache_key]

    # Evict oldest entries if cache is full
    if len(_processor_cache) >= _MAX_CACHE_SIZE:
        oldest_key = next(iter(_processor_cache))
        del _processor_cache[oldest_key]

    # mlx-lm uses TokenizerWrapper which wraps an HF tokenizer in `_tokenizer`.
    # Outlines expects a tokenizer implementing its Tokenizer protocol.
    # TransformerTokenizer adapts HF tokenizers to this protocol.
    base_tokenizer = tokenizer
    if (
        tokenizer.__class__.__name__ == "TokenizerWrapper"
        and hasattr(tokenizer, "_tokenizer")
    ):
        base_tokenizer = tokenizer._tokenizer
    if hasattr(base_tokenizer, "vocabulary") and hasattr(base_tokenizer, "convert_token_to_string"):
        outlines_tokenizer = base_tokenizer
    else:
        outlines_tokenizer = TransformerTokenizer(base_tokenizer)

    # Outlines requires the tensor backend name to map array ops correctly.
    processor = JSONLogitsProcessor(
        schema=schema,
        tokenizer=outlines_tokenizer,
        tensor_library_name="mlx",
    )
    _processor_cache[cache_key] = processor

    logger.info(f"Compiled Outlines JSON processor for schema (cache size: {len(_processor_cache)})")
    return processor


class OutlinesJSONLogitsProcessor:
    """Logits processor that enforces JSON Schema compliance at the token level.

    Wraps Outlines' JSONLogitsProcessor to match the mlx-lm logits processor
    interface: ``(tokens: mx.array, logits: mx.array) -> mx.array``.

    At each generation step, tokens that would violate the JSON schema have
    their logits set to -inf, guaranteeing structurally valid output.

    This follows the same pattern as oMLX's ThinkingBudgetProcessor.

    Args:
        schema: JSON Schema dict to enforce.
        tokenizer: The model tokenizer (HuggingFace-compatible).

    Example OpenAI API usage::

        POST /v1/chat/completions
        {
            "model": "mlx-community/Qwen3.5-27B-4bit",
            "messages": [{"role": "user", "content": "Extract entities"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "entities",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "persons": {"type": "array", "items": {"type": "string"}},
                            "locations": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["persons", "locations"]
                    }
                }
            }
        }
    """

    def __init__(self, schema: dict, tokenizer: Any):
        self._outlines_processor = _get_or_create_outlines_processor(schema, tokenizer)
        self._schema = schema

    def __call__(self, tokens: Any, logits: Any) -> Any:
        """Apply JSON schema constraint to logits.

        Args:
            tokens: Previously generated token IDs (mx.array, 1D).
            logits: Raw logits for next token (mx.array, 1D or 2D).

        Returns:
            Modified logits with invalid tokens masked to -inf.
        """
        import mlx.core as mx

        # Outlines expects 1D logits; mlx-lm may pass 2D (1, vocab_size)
        needs_reshape = logits.ndim > 1
        if needs_reshape:
            original_shape = logits.shape
            logits_1d = logits.flatten()
        else:
            logits_1d = logits

        # Apply Outlines constraint
        logits_1d = self._outlines_processor(tokens, logits_1d)

        if needs_reshape:
            logits = logits_1d.reshape(original_shape)
        else:
            logits = logits_1d

        return logits

    def __repr__(self) -> str:
        schema_name = self._schema.get("title", "unnamed")
        return f"OutlinesJSONLogitsProcessor(schema={schema_name!r})"


def is_outlines_available() -> bool:
    """Check if Outlines is installed and importable."""
    try:
        from outlines.models import TransformerTokenizer  # noqa: F401
        from outlines.processors.structured import JSONLogitsProcessor  # noqa: F401
        return True
    except ImportError:
        return False


def extract_json_schema(response_format: Any) -> Optional[dict]:
    """Extract the JSON Schema dict from an OpenAI-style response_format.

    Handles both ResponseFormat Pydantic model and raw dict formats.
    Returns the schema dict if type is "json_schema", None otherwise.

    Args:
        response_format: ResponseFormat object or dict.

    Returns:
        The JSON Schema dict, or None if not applicable.
    """
    if response_format is None:
        return None

    # Handle Pydantic ResponseFormat model
    if hasattr(response_format, "type"):
        if response_format.type != "json_schema":
            return None
        if hasattr(response_format, "json_schema") and response_format.json_schema:
            schema = getattr(response_format.json_schema, "schema_", None)
            if schema is None:
                # Try dict-style access
                schema = getattr(response_format.json_schema, "schema", None)
            return schema
        return None

    # Handle raw dict
    if isinstance(response_format, dict):
        if response_format.get("type") != "json_schema":
            return None
        js = response_format.get("json_schema", {})
        return js.get("schema")

    return None
