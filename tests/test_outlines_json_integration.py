# SPDX-License-Identifier: Apache-2.0
"""Tests for Outlines JSON-schema constrained decoding integration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import StreamingResponse

import omlx.api.json_logits_processor as json_lp
import omlx.server as server
from omlx.api.openai_models import ChatCompletionRequest
from omlx.api.responses_models import ResponsesRequest
from omlx.engine.vlm import VLMBatchedEngine
from omlx.request import SamplingParams
from omlx.scheduler import Scheduler


class _FakeHttpRequest:
    async def is_disconnected(self) -> bool:
        return False


def _make_generation_output(text: str = '{"ok": true}') -> SimpleNamespace:
    return SimpleNamespace(
        output_text=text,
        text=text,
        prompt_tokens=12,
        completion_tokens=5,
        cached_tokens=0,
        finish_reason="stop",
        tool_calls=None,
    )


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine.model_type = None
    engine.tokenizer = MagicMock()
    engine.count_chat_tokens.return_value = 10
    engine.chat = AsyncMock(return_value=_make_generation_output())
    return engine


@pytest.fixture
def patched_server(monkeypatch):
    """Patch global server helpers so endpoint handlers are unit-testable."""
    state = server.ServerState()
    monkeypatch.setattr(server, "_server_state", state)
    monkeypatch.setattr(server, "resolve_model_id", lambda model: model)
    monkeypatch.setattr(server, "validate_context_window", lambda *_: None)
    monkeypatch.setattr(
        server,
        "get_sampling_params",
        lambda *args, **kwargs: (0.2, 0.9, 0, 1.0, 0.0, 0.0, 0.0, 64),
    )
    metrics = MagicMock()
    metrics.record_request_complete = MagicMock()
    monkeypatch.setattr(server, "get_server_metrics", lambda: metrics)
    monkeypatch.setattr(
        server,
        "extract_tool_calls_with_thinking",
        lambda *args, **kwargs: SimpleNamespace(
            cleaned_text='{"ok": true}',
            tool_calls=None,
            cleaned_thinking=None,
        ),
    )
    return state


def test_scheduler_adds_json_schema_processor(
    mock_model, mock_tokenizer, monkeypatch
):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}

    class DummyProcessor:
        def __init__(self, schema, tokenizer):
            self.schema = schema
            self.tokenizer = tokenizer

    monkeypatch.setattr(json_lp, "is_outlines_available", lambda: True)
    monkeypatch.setattr(json_lp, "OutlinesJSONLogitsProcessor", DummyProcessor)

    _, processors = scheduler._build_sampler_and_processors(
        SamplingParams(json_schema=schema)
    )

    matched = [p for p in processors if isinstance(p, DummyProcessor)]
    assert len(matched) == 1
    assert matched[0].schema == schema


def test_scheduler_json_schema_fallback_on_processor_error(
    mock_model, mock_tokenizer, monkeypatch, caplog
):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

    class BoomProcessor:
        def __init__(self, *args, **kwargs):
            raise ValueError("invalid schema")

    monkeypatch.setattr(json_lp, "is_outlines_available", lambda: True)
    monkeypatch.setattr(json_lp, "OutlinesJSONLogitsProcessor", BoomProcessor)

    with caplog.at_level("WARNING"):
        _, processors = scheduler._build_sampler_and_processors(
            SamplingParams(json_schema={"type": "object"})
        )

    assert all(type(p).__name__ != "BoomProcessor" for p in processors)
    assert "Failed to create Outlines JSON processor" in caplog.text


def test_scheduler_mixed_requests_keep_processor_isolation(
    mock_model, mock_tokenizer, monkeypatch
):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    created = []

    class DummyProcessor:
        def __init__(self, schema, tokenizer):
            created.append(schema)
            self.schema = schema

    monkeypatch.setattr(json_lp, "is_outlines_available", lambda: True)
    monkeypatch.setattr(json_lp, "OutlinesJSONLogitsProcessor", DummyProcessor)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    _, with_schema = scheduler._build_sampler_and_processors(
        SamplingParams(json_schema=schema)
    )
    _, without_schema = scheduler._build_sampler_and_processors(SamplingParams())

    assert sum(isinstance(p, DummyProcessor) for p in with_schema) == 1
    assert sum(isinstance(p, DummyProcessor) for p in without_schema) == 0
    assert created == [schema]


def test_wrapper_preserves_1d_and_2d_logits_shape(monkeypatch):
    mx = pytest.importorskip("mlx.core")

    monkeypatch.setattr(
        json_lp,
        "_get_or_create_outlines_processor",
        lambda schema, tokenizer: (lambda tokens, logits: logits + 1.0),
    )

    proc = json_lp.OutlinesJSONLogitsProcessor(
        schema={"type": "object"}, tokenizer=object()
    )
    one_d = proc(mx.array([1, 2, 3]), mx.zeros((32,)))
    two_d = proc(mx.array([1, 2, 3]), mx.zeros((1, 32)))

    assert one_d.shape == (32,)
    assert two_d.shape == (1, 32)


def test_outlines_processor_cache_hits_same_schema(monkeypatch):
    pytest.importorskip("outlines")
    import outlines.models as outlines_models
    import outlines.processors.structured as outlines_structured

    json_lp._processor_cache.clear()
    calls = {"count": 0}

    class FakeJsonProcessor:
        def __init__(self, schema, tokenizer, tensor_library_name):
            calls["count"] += 1
            self.schema = schema
            self.tokenizer = tokenizer
            self.tensor_library_name = tensor_library_name

        def __call__(self, tokens, logits):
            return logits

    monkeypatch.setattr(outlines_structured, "JSONLogitsProcessor", FakeJsonProcessor)
    monkeypatch.setattr(outlines_models, "TransformerTokenizer", lambda tok: tok)

    tokenizer = SimpleNamespace(
        vocabulary={"{": 0, "}": 1},
        convert_token_to_string=lambda token: token,
    )
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}

    first = json_lp._get_or_create_outlines_processor(schema, tokenizer)
    second = json_lp._get_or_create_outlines_processor(schema, tokenizer)

    assert first is second
    assert calls["count"] == 1
    json_lp._processor_cache.clear()


@pytest.mark.asyncio
async def test_chat_completion_adds_json_schema_when_outlines_available(
    monkeypatch, patched_server
):
    engine = _make_engine()

    async def _get_engine(_model):
        return engine

    monkeypatch.setattr(server, "get_engine_for_model", _get_engine)
    monkeypatch.setattr(json_lp, "is_outlines_available", lambda: True)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Extract entities"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "entities",
                "schema": {
                    "type": "object",
                    "properties": {"persons": {"type": "array"}},
                    "required": ["persons"],
                },
            },
        },
    )

    await server.create_chat_completion(request, _FakeHttpRequest(), True)
    kwargs = engine.chat.await_args.kwargs
    assert kwargs["json_schema"]["type"] == "object"
    assert kwargs["json_schema"]["required"] == ["persons"]


@pytest.mark.asyncio
async def test_chat_completion_omits_json_schema_when_outlines_missing(
    monkeypatch, patched_server
):
    engine = _make_engine()

    async def _get_engine(_model):
        return engine

    monkeypatch.setattr(server, "get_engine_for_model", _get_engine)
    monkeypatch.setattr(json_lp, "is_outlines_available", lambda: False)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Extract entities"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "x", "schema": {"type": "object"}},
        },
    )

    await server.create_chat_completion(request, _FakeHttpRequest(), True)
    kwargs = engine.chat.await_args.kwargs
    assert "json_schema" not in kwargs


@pytest.mark.asyncio
async def test_chat_completion_json_object_does_not_set_json_schema(
    monkeypatch, patched_server
):
    engine = _make_engine()

    async def _get_engine(_model):
        return engine

    monkeypatch.setattr(server, "get_engine_for_model", _get_engine)
    monkeypatch.setattr(json_lp, "is_outlines_available", lambda: True)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Return JSON"}],
        response_format={"type": "json_object"},
    )

    await server.create_chat_completion(request, _FakeHttpRequest(), True)
    kwargs = engine.chat.await_args.kwargs
    assert "json_schema" not in kwargs


@pytest.mark.asyncio
async def test_chat_completion_streaming_forwards_json_schema(
    monkeypatch, patched_server
):
    engine = _make_engine()
    captured = {}

    async def _get_engine(_model):
        return engine

    def _fake_stream(*args, **kwargs):
        captured.update(kwargs)
        async def _gen():
            yield "data: [DONE]\n\n"
        return _gen()

    monkeypatch.setattr(server, "get_engine_for_model", _get_engine)
    monkeypatch.setattr(server, "stream_chat_completion", _fake_stream)
    monkeypatch.setattr(
        server, "_with_sse_keepalive", lambda ait, http_request: ait
    )
    monkeypatch.setattr(json_lp, "is_outlines_available", lambda: True)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Extract entities"}],
        stream=True,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "x", "schema": {"type": "object"}},
        },
    )

    response = await server.create_chat_completion(request, _FakeHttpRequest(), True)
    assert isinstance(response, StreamingResponse)
    assert captured["json_schema"]["type"] == "object"


@pytest.mark.asyncio
async def test_responses_api_streaming_forwards_json_schema(
    monkeypatch, patched_server
):
    engine = _make_engine()
    captured = {}

    async def _get_engine(_model):
        return engine

    def _fake_stream(*args, **kwargs):
        captured.update(kwargs)
        async def _gen():
            yield "data: [DONE]\n\n"
        return _gen()

    monkeypatch.setattr(server, "get_engine_for_model", _get_engine)
    monkeypatch.setattr(server, "stream_responses_api", _fake_stream)
    monkeypatch.setattr(
        server, "_with_sse_keepalive", lambda ait, http_request: ait
    )
    monkeypatch.setattr(json_lp, "is_outlines_available", lambda: True)

    request = ResponsesRequest(
        model="test-model",
        input="Extract entities",
        stream=True,
        text={
            "format": {
                "type": "json_schema",
                "name": "entities",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                "strict": True,
            }
        },
    )

    response = await server.create_response(request, _FakeHttpRequest(), True)
    assert isinstance(response, StreamingResponse)
    assert captured["json_schema"]["type"] == "object"


@pytest.mark.asyncio
async def test_vlm_generate_passes_json_schema_to_sampling_params():
    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True
    engine._engine = MagicMock()
    engine._engine.generate = AsyncMock(return_value=_make_generation_output())

    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    await engine.generate(
        prompt="Describe image",
        vlm_inputs_embeds=MagicMock(),
        vlm_extra_kwargs={"position_ids": [0]},
        vlm_image_hash="img-hash",
        json_schema=schema,
    )

    kwargs = engine._engine.generate.await_args.kwargs
    sampling_params = kwargs["sampling_params"]
    assert sampling_params.json_schema == schema
